from datetime import timedelta

from sqlalchemy import select

from app.models import Annotation, Outbreak
from scripts_import import load_outbreaks


def test_loads_full_csv_idempotently(db):
    stats = load_outbreaks.load(load_outbreaks.DEFAULT_CSV, db)
    assert stats["rows"] == stats["inserted"] == 9962
    assert stats["short_series"] == 228

    again = load_outbreaks.load(load_outbreaks.DEFAULT_CSV, db)
    assert again["inserted"] == again["updated"] == 0
    assert again["unchanged"] == 9962


def test_series_fit_inside_date_window(db):
    load_outbreaks.load(load_outbreaks.DEFAULT_CSV, db)
    for o in db.execute(select(Outbreak)).scalars():
        assert o.values, o.id
        assert o.start_date + timedelta(weeks=len(o.values) - 1) <= o.end_date, o.id
        assert all(v is None or v >= 0 for v in o.values), o.id
    first = db.get(Outbreak, 1)
    assert (first.disease, first.location, len(first.values)) == ("RSV", "ND", 46)


def test_reload_reports_changed_annotated_outbreaks(db, tmp_path):
    src = load_outbreaks.DEFAULT_CSV.read_text().splitlines()
    small = tmp_path / "small.csv"
    small.write_text("\n".join(src[:3]) + "\n")
    load_outbreaks.load(small, db)

    from tests.conftest import make_user
    user = make_user(db, "alice")
    db.add(Annotation(outbreak_id=1, user_id=user.id, label="outbreak", confidence=4))
    db.commit()

    header, row1, row2 = src[:3]
    small.write_text("\n".join([header, row1.replace(",459.0,", ",460.0,"), row2]) + "\n")
    stats = load_outbreaks.load(small, db)
    assert stats["updated"] == 1 and stats["updated_but_annotated"] == [1]
    assert db.execute(select(Annotation)).scalars().one().label == "outbreak"
