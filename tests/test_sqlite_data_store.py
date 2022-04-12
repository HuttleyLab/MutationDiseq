import pathlib

from pathlib import Path

import pytest

from cogent3.app.composable import NotCompleted
from cogent3.app.io import get_data_store

from mdeq.sqlite_data_store import (
    ReadonlySqliteDataStore,
    WriteableSqliteDataStore,
    sql_loader,
    sql_writer,
)
from mdeq.utils import CompressedValue


@pytest.fixture(scope="session")
def tmp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp("sqldb")


@pytest.fixture(scope="session")
def ro_dir_dstore():
    DATADIR = Path("~/repos/Cogent3/tests/data").expanduser()
    return get_data_store(DATADIR, suffix="fasta")


@pytest.fixture(scope="session")
def sql_dstore_path(ro_dir_dstore, tmp_dir):
    # we now need to write these out to a path
    path = tmp_dir / "data.sqlitedb"
    wr = WriteableSqliteDataStore(path, if_exists="overwrite")
    for m in ro_dir_dstore:
        wr.write(data={"data": m.read()}, identifier=f"{Path(m.name).stem}.json")
    return path


@pytest.fixture(scope="session")
def ro_sql_dstore(sql_dstore_path):
    dstore = get_data_store(sql_dstore_path)
    return dstore


def test_sql_dstore_contains(ro_sql_dstore):
    """contains operation works for tinydb data store"""
    for m in ro_sql_dstore:
        assert m in ro_sql_dstore
        n = str(m)
        assert n in ro_sql_dstore

    assert "blag" not in ro_sql_dstore


def test_sql_len(ro_sql_dstore, ro_dir_dstore):
    """len matches directory data store"""
    assert len(ro_sql_dstore) == len(ro_dir_dstore)


def test_sql_get_member(ro_sql_dstore):
    """get member works on TinyDbDataStore"""
    m = ro_sql_dstore.get_member("brca1.json")
    assert m.parent is ro_sql_dstore
    m = ro_sql_dstore.get_member("non-existent")
    assert m is None


def test_sql_iter(ro_sql_dstore, ro_dir_dstore):
    """sql dstore iter works"""
    expect = [f"{Path(m.name).stem}.json" for m in ro_dir_dstore]
    got = [m.name for m in ro_sql_dstore]
    assert set(got) == set(expect)


def test_sql_filter(ro_sql_dstore):
    """filtering members by name"""
    patterns = [("*gans*", 1), ("primate*.json", 2), ("primates_brca1.json", 1)]
    for pattern, count in patterns:
        got = ro_sql_dstore.filtered(pattern)
        assert len(got) == count
        for result in got:
            for part in pattern.split("*"):
                assert part in result


def test_sql_pickleable_roundtrip(ro_sql_dstore):
    """pickling of data stores should be reversible"""
    from pickle import dumps, loads

    expect = [m.name for m in ro_sql_dstore]

    round_tripped = loads(dumps(ro_sql_dstore))
    got = [m.name for m in round_tripped]
    assert got == expect


### write tests
@pytest.fixture()
def rw_sql_dstore_mem(ro_dir_dstore):
    """in memory dstore"""
    db = WriteableSqliteDataStore(":memory:")
    for m in ro_dir_dstore:
        db.write(data={"data": m.read()}, identifier=f"{Path(m.name).stem}.json")
    return db


def test_sql_add_read_log(rw_sql_dstore_mem, tmp_path):
    """adding log from file to tinydb should work"""
    # no log file data yet
    got = rw_sql_dstore_mem.logs[0]
    assert got.deserialised == ""
    log_file = pathlib.Path(tmp_path) / "mylog.log"
    txt = "\n".join(["some text", "on multiple lines"])
    log_file.write_text(txt)
    assert len(rw_sql_dstore_mem.logs) == 1
    got = rw_sql_dstore_mem.add_log(log_file, cleanup=False)
    assert got == 1  # the log id, only one log
    assert log_file.exists()
    got = rw_sql_dstore_mem.logs[0]
    assert got.decompressed == txt.encode("utf8")
    # the following will update the log file but clean up
    got = rw_sql_dstore_mem.add_log(log_file, cleanup=True)
    assert got == 1  # the log id, only one log
    assert not log_file.exists()
    got = rw_sql_dstore_mem.logs[0]
    assert got.decompressed == txt.encode("utf8")


def test_sql_add_read_log_txt(rw_sql_dstore_mem, tmp_path):
    """adding log from file to tinydb should work"""
    # no log file data yet
    got = rw_sql_dstore_mem.logs[0]
    assert got.deserialised == ""
    txt = "\n".join(["some text", "on multiple lines"])
    got = rw_sql_dstore_mem.add_log(txt)
    assert got == 1  # the log id, ojnly one log
    got = rw_sql_dstore_mem.logs[0]
    assert got.decompressed == txt.encode("utf8")


def test_sql_preserves_original_data(rw_sql_dstore_mem):
    """original data is not modified to create record"""
    orig = {"data": "ACGGT", "source": "blah.json"}
    backup = {**orig}
    assert backup is not orig
    assert backup == orig
    rw_sql_dstore_mem.write(data=orig)
    assert backup == orig
    m = rw_sql_dstore_mem.get_member("blah.json")
    data = m.read()
    for k, v in data.items():
        data[k] = v.deserialised
    assert data == backup


def test_sql_rw_incomplete(rw_sql_dstore_mem):
    """write NotCompleted"""

    ncomp = NotCompleted("FAIL", "somefunc", "checking", source="testing.json")
    assert len(rw_sql_dstore_mem.incomplete) == 0
    rw_sql_dstore_mem.write(data=ncomp)
    assert len(rw_sql_dstore_mem.incomplete) == 1
    got = rw_sql_dstore_mem.incomplete[0]
    data = got.read()
    for k, v in data.items():
        data[k] = v.deserialised


def test_sql_describe(rw_sql_dstore_mem):
    """produce a table"""
    from cogent3.util.table import Table

    desc = rw_sql_dstore_mem.describe
    assert isinstance(desc, Table)
    assert desc[1, "number"] == 0
    ncomp = NotCompleted("FAIL", "somefunc", "checking", source="testing.json")
    rw_sql_dstore_mem.write(data=ncomp)
    desc = rw_sql_dstore_mem.describe
    assert desc[1, "number"] == 1


def test_sql_summary_incomplete(rw_sql_dstore_mem):
    """produce a table"""
    from cogent3.util.table import Table

    ncomp = NotCompleted("FAIL", "somefunc", "checking", source="testing.json")
    rw_sql_dstore_mem.write(data=ncomp)
    summary = rw_sql_dstore_mem.summary_incomplete
    assert isinstance(summary, Table)
    assert summary[0, "num"] == 1


def test_sql_dblock(rw_sql_dstore_mem):
    """locking/unlocking of db"""
    from os import getpid

    assert rw_sql_dstore_mem._lock_pid == getpid()

    # unlocking
    rw_sql_dstore_mem.unlock()
    assert rw_sql_dstore_mem._lock_pid is None

    # introduce an artificial lock
    rw_sql_dstore_mem.db.execute("UPDATE state SET lock_pid = 123 where state_id = 1")
    assert rw_sql_dstore_mem._lock_pid == 123
    rw_sql_dstore_mem.unlock()  # has no effect since doesn't match our pid
    assert rw_sql_dstore_mem._lock_pid == 123
    rw_sql_dstore_mem.unlock(force=True)
    assert rw_sql_dstore_mem._lock_pid is None

    # now try to lock
    with pytest.raises(RuntimeError):
        rw_sql_dstore_mem.lock(force=False)

    # force back to current pid
    rw_sql_dstore_mem.lock(force=True)

    # artificially induce another lock record
    # lock() should trigger exception
    rw_sql_dstore_mem.db.execute("INSERT INTO state(lock_pid) VALUES (?)", (123,))
    with pytest.raises(RuntimeError):
        rw_sql_dstore_mem.lock(force=True)


def test_db_creation(sql_dstore_path, tmp_path):
    """overwrite, raise, ignore conditions"""

    with pytest.raises(FileExistsError):
        WriteableSqliteDataStore(sql_dstore_path, if_exists="raise")

    ro_dstore = ReadonlySqliteDataStore(sql_dstore_path)
    num_members = len(ro_dstore)
    assert num_members > 0
    del ro_dstore

    rw_dstore = WriteableSqliteDataStore(sql_dstore_path, if_exists="overwrite")
    assert len(rw_dstore) == 0


def test_sql_loader(rw_sql_dstore_mem):
    loader = sql_loader()
    result = loader(rw_sql_dstore_mem[0])
    print(result.keys())


def test_sql_writer(rw_sql_dstore_mem):
    from cogent3 import make_aligned_seqs
    from cogent3.util.deserialise import deserialise_object

    aln = make_aligned_seqs(
        dict(a="ACGGT", b="AC--T"), moltype="dna", info=dict(source="blah.fasta")
    )
    writer = sql_writer(":memory:")
    result = writer(aln)
    assert result == f"{Path(aln.info.source)}.json"

    loader = sql_loader()
    got = loader(writer.data_store.get_member(result))
    assert got.to_dict() == aln.to_dict()

    # check that if we block the full deserialisation, we have to expand it
    loader = sql_loader(fully_deserialise=False)
    got = loader(writer.data_store.get_member(result))
    for k, v in got.items():
        got[k] = v.deserialised

    obj = deserialise_object(got)
    assert obj.to_dict() == aln.to_dict()
