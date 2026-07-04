import io
import unittest

from xquik_import import XquikImportError, load_xquik_texts


def named_upload(name, content):
    upload = io.BytesIO(content.encode("utf-8"))
    upload.name = name
    return upload


class XquikImportTests(unittest.TestCase):
    def test_loads_csv_text_alias(self):
        self.assertEqual(
            load_xquik_texts(named_upload("tweets.csv", "tweet_text\nGreat launch\n")),
            ["Great launch"],
        )

    def test_loads_json_results_array(self):
        self.assertEqual(
            load_xquik_texts(
                named_upload("tweets.json", '{"results":[{"full_text":"Useful update"}]}')
            ),
            ["Useful update"],
        )

    def test_loads_nested_tweet_text(self):
        self.assertEqual(
            load_xquik_texts(
                named_upload("tweets.jsonl", '{"tweet":{"text":"Nested export"}}\n')
            ),
            ["Nested export"],
        )

    def test_rejects_missing_text_fields(self):
        with self.assertRaises(XquikImportError):
            load_xquik_texts(named_upload("tweets.json", '{"results":[{"id":"1"}]}'))


if __name__ == "__main__":
    unittest.main()
