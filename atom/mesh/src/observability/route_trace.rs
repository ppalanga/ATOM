//! Per-request routing trace behind the sampled `Selected worker` log.
//!
//! Inference routes run inside [`scope`]. Policies select and routers dispatch
//! on the request task before the response is returned, so both reach the
//! trace through a task-local instead of the placement API.

use std::sync::{Arc, OnceLock};

use axum::{extract::Request, http::HeaderMap, middleware::Next, response::Response};
use parking_lot::Mutex;
use rand::Rng;
use tracing::info;

use crate::{core::Worker, observability::metrics::MeshMetrics, policies::decision::RouteDecision};

/// Request header that logs the routing decisions of one request whatever the sample rate.
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
struct RouteTrace {
    log: bool,
    /// Latest policy selection per pool; retries replace it.
    selections: Mutex<Vec<Selection>>,
}

struct Selection {
    pool: &'static str,
    policy: &'static str,
    worker: Arc<dyn Worker>,
    decision: RouteDecision,
}

fn should_log(headers: &HeaderMap, rate: f64) -> bool {
    headers.contains_key(ROUTE_TRACE_HEADER) || rand::rng().random::<f64>() < rate
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
            decision,
        });
    });
}

/// Records the worker a request is sent to after rank mapping.
pub(crate) fn record_dispatch(pool: &'static str, worker: &Arc<dyn Worker>) {
    MeshMetrics::record_selected_dp_rank(pool, worker.base_url(), worker.dp_rank());
    let _ = TRACE.try_with(|trace| {
        if !trace.log {
            return;
        }
        let selections = trace.selections.lock();
        let selection = selections.iter().find(|selection| selection.pool == pool);
        let decision = selection.map(|selection| selection.decision);
        info!(
            target: "mesh::route",
            pool,
            policy = selection.map(|selection| selection.policy),
            branch = decision.map(|decision| decision.branch),
            worker = worker.base_url(),
            dp_rank = worker.dp_rank(),
            remapped_from = selection
                .map(|selection| selection.worker.url())
                .filter(|url| *url != worker.url()),
            match_ratio = decision.and_then(|decision| decision.match_ratio),
            cost = decision.and_then(|decision| decision.cost),
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
    use crate::core::{BasicWorkerBuilder, DPAwareWorkerBuilder, WorkerType};

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
    fn header_forces_the_log_and_rate_samples_the_rest() {
        let mut headers = HeaderMap::new();
        assert!(!should_log(&headers, 0.0));
        assert!(should_log(&headers, 1.0));
        headers.insert(ROUTE_TRACE_HEADER, "1".parse().unwrap());
        assert!(should_log(&headers, 0.0));
    }

    #[tokio::test]
    async fn scope_installs_the_trace_for_the_handler() {
        let app = Router::new()
            .route(
                "/",
                get(|| async { TRACE.with(|trace| trace.log.to_string()) }),
            )
            .layer(axum::middleware::from_fn(scope));
        let request = Request::builder()
            .uri("/")
            .header(ROUTE_TRACE_HEADER, "1")
            .body(Body::empty())
            .unwrap();

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
}
