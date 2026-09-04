"""Manual smoke test for pdf_processor.py.

Pass a PDF path as the first argument, or place one PDF in the project root.
"""

from __future__ import annotations

import sys
from pathlib import Path

from pdf_processor import PDFProcessingError, process_pdf


def _find_test_pdf() -> Path | None:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    project_root = Path(__file__).resolve().parent
    return next(project_root.glob("*.pdf"), None)


def main() -> int:
    pdf_path = _find_test_pdf()
    if pdf_path is None:
        print("项目目录中没有可用于测试的 PDF，请放入一个可复制文字的 PDF 后重新运行。")
        return 1
    if not pdf_path.is_file():
        print(f"测试 PDF 不存在：{pdf_path}")
        return 1

    try:
        result = process_pdf(pdf_path)
    except PDFProcessingError as exc:
        print(str(exc))
        return 1
    except Exception as exc:
        print(f"PDF 处理失败：{type(exc).__name__}: {exc}")
        return 1

    print(f"PDF 文件名：{result['source_file']}")
    print(f"document_id 前 12 位：{result['document_id'][:12]}")
    print(f"总页数：{result['total_pages']}")
    print(f"chunk 总数：{len(result['chunks'])}")
    for index, chunk in enumerate(result["chunks"][:3], start=1):
        preview = chunk["chunk_text"][:100].replace("\n", " ")
        print(
            f"Chunk {index}：页码={chunk['page_number']}，"
            f"字符数={len(chunk['chunk_text'])}，前 100 字={preview}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

