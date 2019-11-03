"""
    This module is dedicated to perform the essay evaluation
    Before running this file, make sure the server to use the cogroo api is running in your computer.
    The instructions to run the CoGroo server can be found in the website https://github.com/gpassero/cogroo4py
"""
import numpy as np
from cogroo_interface import Cogroo
from data_procedures import get_tfidf_of_essays

cogroo = Cogroo.Instance()

def create_rules_id_dictionary(n_rules=129):
    """
    Creates a dictionary for the rules returned by the CoGroo API.
    This dictionary is going to be used for when the rule is retuned for the API,
    its id will come in the format: "xml:<number>", So based on the number stored in the corresponding
    position of the id in format "xml:<number>" the value in the index of the rule in the feature vector
    will be increased.

    :param n_rules: number of rules
    :return:
    """
    prefix = "xml:"
    rules = {}
    rules_vector = np.zeros((n_rules,))
    for i in range(n_rules):
        rules[prefix+str(i+1)] = i
    return rules, rules_vector


def main():
    rules_dict, rule_vector= create_rules_id_dictionary()
    print("Rules dict")
    print(rules_dict)
    tfidfs = get_tfidf_of_essays()
    print(tfidfs.shape)
    # doc = cogroo.grammar_check(essay.text)
    # mistakes = doc.mistakes
    """
    for m in mistakes:
        if 'space' not in m.rule_id:
            rule_index = rules_dict[m.rule_id]
            print(rule_index)
    """

if __name__ == "__main__":
    main()
