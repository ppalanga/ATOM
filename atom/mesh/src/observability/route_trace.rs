//! Per-request routing trace behind the sampled `Selected worker` log and the
//! reconciliation of the router's cache estimate with the reported usage.
//!
//! Inference routes run inside [`scope`]. Policies select and routers dispatch
//! on the request task before the response is returned, so both reach the
//! trace through a task-local instead of the placement API.

use std::sync::{Arc, OnceLock};

use axum::{extract::Request, http::HeaderMap, middleware::Next, response::Response};
use parking_lot::Mutex;
use rand::Rng;
use tracing::info;

use crate::{
    core::Worker,
    observability::metrics::{metrics_labels, MeshMetrics},
    policies::decision::RouteDecision,
};

/// Request header that logs the routing decisions of one request while the
/// sample rate is above 0.
pub const ROUTE_TRACE_HEADER: &str = "x-mesh-route-trace";

static SAMPLE_RATE: OnceLock<f64> = OnceLock::new();

tokio::task_local! {
    static TRACE: Arc<RouteTrace>;
}

/// Fraction of inference requests that log their routing decisions. Only the
/// first call takes effect.
pub fn set_sample_rate(rate: f64) {
    let _ = SAMPLE_RATE.set(rate);
}

#[derive(Default)]
pub(crate) struct RouteTrace {
    log: bool,
    /// Latest policy selection per pool; retries replace it.
    selections: Mutex<Vec<Selection>>,
}

impl RouteTrace {
    /// Compares the cached prompt tokens a response reported with the router's
    /// estimate on the pool that served the prompt.
    pub(crate) fn reconcile(&self, prompt: u64, cached: u64) {
        let selections = self.selections.lock();
        let Some(selection) = selections.iter().find(|selection| {
            matches!(
                selection.pool,
                metrics_labels::WORKER_PREFILL | metrics_labels::WORKER_REGULAR
            )
        }) else {
            return;
        };
        MeshMetrics::record_route_cache_tokens(selection.pool, prompt, cached);
        if let Some(ratio) = selection.decision.estimated_hit_ratio {
            let estimated = (ratio * prompt as f64).round() as u64;
            MeshMetrics::record_route_cache_estimate(selection.pool, prompt, estimated, cached);
        }
    }
}

/// The current request's trace, for observers that outlive the handler.
pub(crate) fn current() -> Option<Arc<RouteTrace>> {
    TRACE.try_with(Arc::clone).ok()
}

struct Selection {
    pool: &'static str,
    policy: &'static str,
    worker: Arc<dyn Worker>,
    /// Router-tracked active requests on `worker` when the policy selected it.
    active_requests: usize,
    decision: RouteDecision,
}

/// Clients can send the header, so it only applies once the operator enables sampling.
fn should_log(headers: &HeaderMap, rate: f64) -> bool {
    rate > 0.0 && (headers.contains_key(ROUTE_TRACE_HEADER) || rand::rng().random::<f64>() < rate)
}

pub(crate) async fn scope(request: Request, next: Next) -> Response {
    let rate = SAMPLE_RATE.get().copied().unwrap_or(0.0);
    let trace = RouteTrace {
        log: should_log(request.headers(), rate),
        ..RouteTrace::default()
    };
    TRACE.scope(Arc::new(trace), next.run(request)).await
}

/// Keeps the decision of a policy selection until the router dispatches it.
pub(crate) fn record_selection(
    pool: &'static str,
    policy: &'static str,
    worker: Option<&Arc<dyn Worker>>,
    decision: RouteDecision,
) {
    let _ = TRACE.try_with(|trace| {
        let Some(worker) = worker else {
            if trace.log {
                info!(target: "mesh::route", pool, policy, branch = decision.branch, "No worker selected");
            }
            return;
        };
        let mut selections = trace.selections.lock();
        selections.retain(|selection| selection.pool != pool);
        selections.push(Selection {
            pool,
            policy,
            worker: worker.clone(),
            active_requests: worker.load(),
            decision,
        });
    });
}

/// Records the worker a request is sent to after rank mapping.
pub(crate) fn record_dispatch(pool: &'static str, worker: &Arc<dyn Worker>) {
    MeshMetrics::record_selected_dp_rank(pool, worker.base_url(), worker.dp_rank());
    let _ = TRACE.try_with(|trace| {
        let mut selections = trace.selections.lock();
        let selection = selections
            .iter_mut()
            .find(|selection| selection.pool == pool);
        let remapped = selection
            .as_ref()
            .is_some_and(|selection| selection.worker.url() != worker.url());
        let selection = selection.map(|selection| {
            if remapped {
                // DP ranks keep separate KV caches, so the estimate does not carry over.
                selection.decision.estimated_hit_ratio = None;
            }
            &*selection
        });
        if !trace.log {
            return;
        }
        let decision = selection.map(|selection| selection.decision);
        info!(
            target: "mesh::route",
            pool,
            policy = selection.map(|selection| selection.policy),
            branch = decision.map(|decision| decision.branch),
            worker = worker.base_url(),
            dp_rank = worker.dp_rank(),
            remapped_from = selection
                .filter(|_| remapped)
                .map(|selection| selection.worker.url()),
            match_ratio = decision.and_then(|decision| decision.match_ratio),
            active_requests = selection.map(|selection| selection.active_requests),
            affinity = decision.and_then(|decision| decision.affinity),
            "Selected worker"
        );
    });
}

#[cfg(test)]
mod tests {
    use std::io;

    use axum::{body::Body, routing::get, Router};
    use tower::ServiceExt;

    use super::*;
    use crate::{
        core::{BasicWorkerBuilder, DPAwareWorkerBuilder, WorkerType},
        observability::request::RequestMetrics,
    };

    #[derive(Clone, Default)]
    struct Capture(Arc<std::sync::Mutex<Vec<u8>>>);

    impl io::Write for Capture {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for Capture {
        type Writer = Self;

        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    impl Capture {
        fn text(&self) -> String {
            String::from_utf8(self.0.lock().unwrap().clone()).unwrap()
        }
    }

    fn prefill_rank(rank: usize) -> Arc<dyn Worker> {
        Arc::new(
            DPAwareWorkerBuilder::new("http://prefill:8000", rank, 2)
                .worker_type(WorkerType::Prefill {
                    bootstrap_port: None,
                })
                .build(),
        )
    }

    async fn traced(log: bool, run: impl FnOnce()) -> String {
        let capture = Capture::default();
        let subscriber = tracing_subscriber::fmt()
            .with_writer(capture.clone())
            .with_ansi(false)
            .finish();
        let _subscriber = tracing::subscriber::set_default(subscriber);
        let trace = RouteTrace {
            log,
            ..RouteTrace::default()
        };
        TRACE.scope(Arc::new(trace), async { run() }).await;
        capture.text()
    }

    #[test]
    fn header_forces_the_log_only_while_sampling() {
        let mut headers = HeaderMap::new();
        assert!(!should_log(&headers, 0.0));
        assert!(should_log(&headers, 1.0));
        headers.insert(ROUTE_TRACE_HEADER, "1".parse().unwrap());
        assert!(!should_log(&headers, 0.0));
        assert!(should_log(&headers, f64::MIN_POSITIVE));
    }

    #[tokio::test]
    async fn scope_installs_the_trace_for_the_handler() {
        let app = Router::new()
            .route("/", get(|| async { current().is_some().to_string() }))
            .layer(axum::middleware::from_fn(scope));
        let request = Request::builder().uri("/").body(Body::empty()).unwrap();

        let response = app.oneshot(request).await.unwrap();
        let body = axum::body::to_bytes(response.into_body(), 64)
            .await
            .unwrap();
        assert_eq!(&body[..], b"true");
    }

    #[test]
    fn dispatch_counts_the_rank_after_mapping() {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let regular: Arc<dyn Worker> = Arc::new(
            BasicWorkerBuilder::new("http://regular:8000")
                .worker_type(WorkerType::Regular)
                .build(),
        );

        metrics::with_local_recorder(&recorder, || {
            record_dispatch("prefill", &prefill_rank(1));
            record_dispatch("regular", &regular);
        });

        let rendered = handle.render();
        for line in [
            r#"mesh_selected_dp_rank_total{pool="prefill",worker="http://prefill:8000",dp_rank="1"} 1"#,
            r#"mesh_selected_dp_rank_total{pool="regular",worker="http://regular:8000",dp_rank="none"} 1"#,
        ] {
            assert!(rendered.contains(line), "missing {line} in\n{rendered}");
        }
    }

    #[tokio::test]
    async fn traced_dispatch_logs_the_decision_and_remap() {
        let run = || {
            let decision = RouteDecision {
                match_ratio: Some(0.75),
                ..RouteDecision::branch("affinity")
            };
            record_selection("prefill", "cache_aware", Some(&prefill_rank(0)), decision);
            record_dispatch("prefill", &prefill_rank(1));
        };

        let logs = traced(true, run).await;
        for field in [
            "Selected worker",
            r#"pool="prefill""#,
            r#"policy="cache_aware""#,
            r#"branch="affinity""#,
            r#"worker="http://prefill:8000""#,
            "dp_rank=1",
            r#"remapped_from="http://prefill:8000@0""#,
            "match_ratio=0.75",
        ] {
            assert!(logs.contains(field), "missing {field} in\n{logs}");
        }
        assert!(traced(false, run).await.is_empty());
    }

    #[tokio::test]
    async fn reported_cached_tokens_are_reconciled_with_the_dispatched_estimate() {
        let recorder = metrics_exporter_prometheus::PrometheusBuilder::new().build_recorder();
        let handle = recorder.handle();
        let _recorder = metrics::set_default_local_recorder(&recorder);
        let body = r#"{"usage":{"prompt_tokens":100,"completion_tokens":1,"prompt_tokens_details":{"cached_tokens":40}}}"#;

        // The second request is remapped to another rank, which drops the estimate.
        for dispatched_rank in [0, 1] {
            let request = async {
                let decision = RouteDecision {
                    estimated_hit_ratio: Some(0.5),
                    ..RouteDecision::branch("affinity")
                };
                record_selection("prefill", "cache_aware", Some(&prefill_rank(0)), decision);
                record_dispatch("prefill", &prefill_rank(dispatched_rank));
                RequestMetrics::new("http", "pd", "m", "/v1/chat/completions", false)
                    .wrap_response(Response::new(Body::from(body)))
            };
            let response = TRACE.scope(Arc::default(), request).await;
            axum::body::to_bytes(response.into_body(), 4096)
                .await
                .unwrap();
        }

        let rendered = handle.render();
        for line in [
            r#"mesh_route_cache_tokens_total{pool="prefill",kind="prompt"} 200"#,
            r#"mesh_route_cache_tokens_total{pool="prefill",kind="cached"} 80"#,
            r#"mesh_route_cache_estimate_tokens_total{pool="prefill",kind="prompt"} 100"#,
            r#"mesh_route_cache_estimate_tokens_total{pool="prefill",kind="estimated"} 50"#,
            r#"mesh_route_cache_estimate_tokens_total{pool="prefill",kind="actual"} 40"#,
            r#"mesh_route_cache_estimate_error_ratio_count{pool="prefill"} 1"#,
        ] {
            assert!(rendered.contains(line), "missing {line} in\n{rendered}");
        }
    }
}
