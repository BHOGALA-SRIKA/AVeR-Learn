"""Tests for reading Bhavya's tree.json as sections (src/indexing/tree_sections.py)."""

from src.indexing.tree_sections import sections_from_tree


def _tree(rows):
    """rows: (kind, text) in reading order -> a minimal tree.json dict."""
    nodes = {"D": {"id": "D", "kind": "document", "title": "doc", "doc_order": 0},
             "C": {"id": "C", "kind": "chapter", "title": "Unit 1 - CPU Scheduling", "doc_order": 1,
                   "page_start": 1, "page_end": 1}}
    for i, (kind, text) in enumerate(rows, 2):
        nid = f"N{i:03d}"
        nodes[nid] = {"id": nid, "kind": kind, "doc_order": i, "page_start": 1 + i // 10, "page_end": 1 + i // 10,
                      "parent_id": "C", ("title" if kind != "paragraph" else "text"): text}
    return {"source_file": "notes.pdf", "nodes": nodes}


def test_bullet_marks_after_a_line_make_it_a_bullet():
    secs = sections_from_tree(_tree([
        ("section", "Dispatcher"),
        ("paragraph", "The dispatcher is a module that:"),
        ("paragraph", "Gives control of the CPU to the selected process"),
        ("paragraph", "•"),
    ]))
    assert [s.heading for s in secs] == ["Dispatcher"]
    assert secs[0].body.splitlines() == ["The dispatcher is a module that:",
                                         "• Gives control of the CPU to the selected process"]


def test_bold_bullet_items_are_not_new_sections():
    secs = sections_from_tree(_tree([
        ("section", "Dispatcher Functions"),
        ("section", "Performs context switch"), ("paragraph", "•"),
        ("section", "Switches CPU to user mode"), ("paragraph", "•"),
        ("section", "Dispatch Latency"),
        ("paragraph", "Time taken to:"), ("paragraph", "•"),
        ("paragraph", "o Stop one process"),
    ]))
    assert [s.heading for s in secs] == ["Dispatcher Functions", "Dispatch Latency"]
    assert secs[0].body.splitlines() == ["• Performs context switch", "• Switches CPU to user mode"]
    assert secs[1].body.splitlines() == ["• Time taken to:", "o Stop one process"]


def test_wrapped_sentence_and_colon_lines_are_not_headings_and_page_numbers_are_dropped():
    secs = sections_from_tree(_tree([
        ("section", "Basic Concepts"),
        ("section", "In a system with one core, only one process can run at a time. All"),
        ("paragraph", "other processes must wait."),
        ("section", "On a multicore system, the same concept is extended:"),
        ("paragraph", "4"),
        ("paragraph", "Each core should be kept busy"), ("paragraph", "•"),
    ]))
    assert [s.heading for s in secs] == ["Basic Concepts"]
    assert secs[0].body.splitlines() == [
        "In a system with one core, only one process can run at a time. All other processes must wait.",
        "On a multicore system, the same concept is extended:",
        "• Each core should be kept busy"]


def test_sections_keep_source_pages_and_node_ids():
    secs = sections_from_tree(_tree([("section", "Dispatcher"), ("paragraph", "Gives control of the CPU")]))
    s = secs[0]
    assert s.pid == "N002" and s.chapter == "Unit 1 - CPU Scheduling"
    assert s.meta["source_pdf"] == "notes.pdf" and s.meta["page_start"] == 1
    assert s.meta["node_ids"] == ["N002", "N003"]
