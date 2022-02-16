
class AspectDatasetColumns:
    input_column = 'text_tokens'
    annotation_column = 'annotations'
    label_column = 'tags'
    aspect_flag_column = 'type_aspect'
    text_column = 'review_text'


class AspectDatasetFields:
    empty_tokens = 'NULL'
    annotations_tricky = [['implicit_aspect'], ['corrupted'], ['aspect_multi_occurrence']]
    tags = {"non_aspect": "O", "beginning_aspect": "B", "inside_aspect": "I"}
    aspect_flag_true = "aspect"
    aspect_flag_false = "non_aspect"
    tags_decoder = {0: "O", 1: "B", 2: "I"}
    tags_encoder = {"O": 0, "B": 1, "I": 2}
    label_list = ["O", "B", "I"]
