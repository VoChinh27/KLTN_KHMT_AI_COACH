from gui_app.services.db_service import DbService


def test_schema_and_update(tmp_path):
    db = DbService(tmp_path / "t.db")
    assert db.list_sessions() == []
    db.update_paths(1, output_video_path="x")      # không lỗi khi chưa có bản ghi
