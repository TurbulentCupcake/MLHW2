f feature_node.isRoot() == True:

        if feature_node.getFeatureName() != None:
            if metadata[feature_node.getFeatureName()][0] == 'numeric':
                if feature_node.left == True:
                    sign = '<='
                    if feature_node.leaf == True:
                        print(level * '|\t' + '{} {} {} {}: {}'.format(feature_node.getFeatureName(), sign,
                                                                       feature_node.getFeatureType(),
                                                                       feature_node.get_label_count(),
                                                                       feature_node.get_label_name()))
                    else:
                        print(level * '|\t' + '{} {} {} {}'.format(feature_node.getFeatureName(), sign,
                                                                   feature_node.getFeatureType(),
                                                                   feature_node.get_label_count()))
                else:
                    sign = '>'
                    if feature_node.leaf == True:
                        print(level * '|\t' + '{} {} {} {}: {}'.format(feature_node.getFeatureName(), sign,
                                                                       feature_node.getFeatureType(),
                                                                       feature_node.get_label_count(),
                                                                       feature_node.get_label_name()))
                    else:
                        print(level * '|\t' + '{} {} {} {}'.format(feature_node.getFeatureName(), sign,
                                                                   feature_node.getFeatureType(),
                                                                   feature_node.get_label_count()))
            else:
                sign = '='
                if feature_node.leaf == True:
                    print(level * '|\t' + '{} {} {} {}: {}'.format(feature_node.getFeatureName(), sign,
                                                                   feature_node.getFeatureType(),
                                                                   feature_node.get_label_count(),
                                                                   feature_node.get_label_name()))
                else:
                    print(level * '|\t' + '{} {} {} {}'.format(feature_node.getFeatureName(), sign,
                                                               feature_node.getFeatureType(),
                                                               feature_node.get_label_count()))

        level = level + 1