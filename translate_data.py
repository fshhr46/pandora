import dl_translate as dlt
import json


def main():
    file_path = "/home/haoranhuang/workspace/pandora/test_data/poseidon_real.json"
    dataset = json.load(open(file_path))
    columns = dataset["column_data"]

    mt = dlt.TranslationModel()
    translations = {}
    for col_id, column in columns.items():
        column_comment = column["metadata"]["comment"]
        translation = mt.translate(
            column_comment, source=dlt.lang.CHINESE, target=dlt.lang.ENGLISH)
        print(f"{column_comment} -> {translation}")
        translations[column_comment] = translation
        dataset["column_data"][col_id]["metadata"]["column_name"] = translation
        dataset["column_data"][col_id]["column_name"] = translation

    output_file_path = "/home/haoranhuang/workspace/pandora/test_data/poseidon_real_translated.json"
    with open(output_file_path, "w") as f:
        json.dump(dataset, f, ensure_ascii=False)

    # translation_file_path = "/home/haoranhuang/workspace/pandora/test_data/translations.json"
    # home = str(Path.home())
    # translation_file_path = os.path.join(
    #     home, "workspace/pandora/test_data/translations.json")
    # with open(translation_file_path, "w") as f:
    #     json.dump(translations, f, ensure_ascii=False)


if __name__ == '__main__':
    main()
