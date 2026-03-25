"""Single-row classifier entry point."""

from dataclasses import dataclass

import polars as pl

from linkages.build_query import build_query
from linkages.translator import detect_language, translate_eng


@dataclass
class TypeClassifier:
    """Run the first pipeline stages on one text string at a time."""

    def prepare_text(self, text: str) -> dict[str, str]:
        detected_lang = detect_language(text)
        english_text = translate_eng(text, from_lang=detected_lang)
        return {
            "input_text": text,
            "detected_language": detected_lang,
            "english_text": english_text,
        }

    def classify(self, text: str) -> dict[str, str]:
        return self.prepare_text(text)

    def classify_row(self, row: dict) -> dict[str, str]:
        query_input = build_query(row)
        result = self.classify(query_input.query)
        result["context"] = query_input.context
        return result


def load_sample_row(
    csv_path: str = "data/raw/ecuador_sample.csv",
    row_index: int = 0,
) -> dict:
    data = pl.read_csv(csv_path)
    if row_index < 0 or row_index >= data.height:
        raise IndexError(f"Row {row_index} is out of range for {csv_path}")
    return data.row(row_index, named=True)


def main() -> None:
    clf = TypeClassifier()
    row = load_sample_row(row_index=1)
    print(clf.classify_row(row))


if __name__ == "__main__":
    main()
