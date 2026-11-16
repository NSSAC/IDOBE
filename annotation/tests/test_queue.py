from sqlalchemy import select

from app.models import Annotation, User
from app.queue import next_outbreak_id, queue_size
from tests.conftest import logged_in_client, make_user


def label(db, user, outbreak_id, value="outbreak"):
    db.add(Annotation(outbreak_id=outbreak_id, user_id=user.id, label=value,
                      confidence=None if value == "skip" else 3))
    db.commit()


def test_never_serves_a_series_twice_and_drains(db, outbreaks):
    client = logged_in_client(db, "alice")
    seen = []
    while True:
        data = client.get("/api/next").json()
        if data["done"]:
            break
        oid = data["outbreak"]["id"]
        assert oid not in seen
        seen.append(oid)
        client.post("/api/annotations", json={"outbreak_id": oid, "label": "skip" if oid == 1 else "unsure", "confidence": 3})
    assert sorted(seen) == [1, 2, 3, 4, 5]
    assert queue_size(db, db.execute(select(User).where(User.username == "alice")).scalar_one().id, target=2) == 0


def test_stops_at_target_and_skips_do_not_count(db, outbreaks):
    a, b, c = (make_user(db, n) for n in "abc")
    label(db, a, 1); label(db, b, 1)               # outbreak 1 reaches target (2)
    label(db, a, 2); label(db, b, 2, "skip")       # outbreak 2 has one real label
    assert queue_size(db, c.id, target=2) == 4      # 2,3,4,5 still open for c
    for _ in range(20):
        assert next_outbreak_id(db, c.id, target=2) != 1


def test_least_covered_first(db, outbreaks):
    a, b, c = (make_user(db, n) for n in "abc")
    for oid in (1, 2, 3, 4):
        label(db, a, oid)
    # Only outbreak 5 has no labels, so c must get it first.
    for _ in range(10):
        assert next_outbreak_id(db, c.id, target=2, order="spread") == 5
    # Changing an answer to skip frees the slot again.
    label(db, a, 4, "skip")
    assert next_outbreak_id(db, b.id, target=2, order="spread") in (4, 5)


def test_complete_order_finishes_series_first(db, outbreaks):
    a, b, c = (make_user(db, n) for n in "abc")
    label(db, a, 3)
    # Outbreak 3 is the only one with a label, so "complete" order serves it to b first.
    for _ in range(10):
        assert next_outbreak_id(db, b.id, target=2, order="complete") == 3
    label(db, b, 3)                  # now at target: no longer served
    for _ in range(10):
        assert next_outbreak_id(db, c.id, target=2, order="complete") != 3


def test_random_order_spans_eligible_series(db, outbreaks):
    a, c = make_user(db, "a"), make_user(db, "c")
    label(db, a, 5)                                 # 5 has one label but is still eligible at target 2
    seen = {next_outbreak_id(db, c.id, target=2, order="random") for _ in range(200)}
    assert seen == {1, 2, 3, 4, 5}
