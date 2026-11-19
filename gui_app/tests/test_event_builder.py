from types import SimpleNamespace as NS

from gui_app.services.event_builder import build_events


def _item(sym, et, name="Góc gối trước"):
    return NS(status_symbol=sym, error_type=NS(value=et), level="PRIMARY", display_name=name,
              student_value=147.0, reference_center=90.0, unit="°", message="Hạ hông")


def _result(items):
    fb = NS(items=items)
    sp = NS(has_score=lambda: True, result=NS(feedback=fb, bundle=NS(static_summary=NS(segment_start_frame=10, segment_end_frame=30))))
    return NS(sub_phase_order=["static"], sub_phase_results={"static": sp})


def test_only_technique_error_pauses():
    ev, q = build_events(_result([_item("✗", "technique_error"), _item("✗", "measurement_error", "Cổ chân")]), {"static": (10, 30)})
    assert len(ev) == 1 and ev[0].frame == 20 and len(ev[0].items) == 1 and len(q) == 1


def test_warn_excluded_by_default():
    ev, _ = build_events(_result([_item("⚠", "technique_error")]), {"static": (10, 30)})
    assert ev == []
    ev, _ = build_events(_result([_item("⚠", "technique_error")]), {"static": (10, 30)}, include_warn=True)
    assert len(ev) == 1
