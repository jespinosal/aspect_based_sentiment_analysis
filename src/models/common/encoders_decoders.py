class LabelsMgr:
    """
    Create labels map to encode and decode labels.
    """

    def __init__(self, unique_labels=None, labels_map=None):

        self.unique_labels = unique_labels
        self.label_id_map = None
        self.label_name_map = labels_map
        self.__set_attributes()

    def __len__(self):
        return len(self.unique_labels)

    def __set_attributes(self):
        if self.label_name_map is not None:
            self.label_id_map, self.unique_labels = self.__load_labels_map()
        elif self.unique_labels is not None:
            self.label_id_map, self.label_name_map = self.__build_labels_map()
        else:
            raise ValueError

    def __build_labels_map(self):
        label_id_map = {label_id: label_name for label_id, label_name in enumerate(self.unique_labels)}
        label_name_map = {label_name: label_id for label_id, label_name in enumerate(self.unique_labels)}
        return label_id_map, label_name_map

    def encoder(self, label_names):
        return [self.label_name_map[label_name] for label_name in label_names]

    def decoder(self, label_ids):
        return [self.label_id_map[label_id] for label_id in label_ids]

    def __load_labels_map(self):
        label_id_map = {label_id: label_name for label_name, label_id in self.label_name_map.items()}
        unique_labels = [label_id_map[label_id] for label_id in range(0, len(self.label_name_map))]
        return label_id_map, unique_labels

    @staticmethod
    def save_decoder(path):
        pass


if __name__ == "__main__":

    # Use case 1: create labels
    label_mgr = LabelsMgr(unique_labels=['a', 'b', 'c', 'd'])
    label_mgr.decoder([1, 2, 3, 0])  # --> ['b', 'c', 'd', 'a']
    label_mgr.encoder(['b', 'c', 'd', 'a'])  # -->[1, 2, 3, 0]

    # use case 2: load labels
    label_mgr = LabelsMgr(labels_map={'a': 0, 'b': 1, 'c': 2, 'd': 3})
    label_mgr.decoder([1, 2, 3, 0])  # --> ['b', 'c', 'd', 'a']
    label_mgr.encoder(['b', 'c', 'd', 'a'])  # -->[1, 2, 3, 0]
