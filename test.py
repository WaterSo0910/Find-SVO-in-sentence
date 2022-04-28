import unittest
from main import *


class TestInverse(unittest.TestCase):
    def test_Rule(self):
        sent = "If Calcavecchia plays Friday morning , he knows the importance of starting well , because the Americans have a recent history of falling behind early ."
        SVOs = [
            [True, "he", "knows", "the importance"],
            [True, "the Americans", "have",
                "a recent history of falling behind early ."],
            [False, "Calcavecchia", "plays", "Friday morning"],
        ]
        for valid, S, V, O in SVOs:
            self.assertEqual(valid, RulesCheck(S, V, O, sent), (S, V, O))

    def test_passive(self):
        sent = "I am drawn to her."
        SVOs = [
            [True, "I", "am drawn", "her"],
            [True, "I", "drawn to", "her"],
            [True, "I", "am drawn to", "her"],
        ]
        for valid, S, V, O in SVOs:
            self.assertEqual(valid, RulesCheck(S, V, O, sent), (S, V, O))

    def test_subject_conj(self):
        sent = "Namin and her sister and her brother are drawn to Alan."
        SVOs = [
            [True, "Namin", "are drawn", "Alan"],
            [True, "her sister", "drawn to", "Alan"],
            [True, "Namin and her sister", "are drawn to", "Alan"],
            [True, "her brother", "are drawn to", "Alan"],
        ]
        for valid, S, V, O in SVOs:
            self.assertEqual(valid, RulesCheck(S, V, O, sent), (S, V, O))

    def test_object_conj(self):
        sent = "Namin is drawn to Alan and his sister and his brother."
        SVOs = [
            [True, "Namin", "is drawn", "Alan"],
            [True, "Namin", "drawn to", "his sister"],
            [True, "Namin", "is drawn to", "his brother"],
        ]
        for valid, S, V, O in SVOs:
            self.assertEqual(valid, RulesCheck(S, V, O, sent), (S, V, O))

    def test_double_verb(self):
        sent = "She kissed and hugged me."
        SVOs = [
            [True, "She", "kissed", "me"],
            [True, "She", "hugged", "me"],
            [True, "She", "kissed and hugged", "me"],
        ]
        for valid, S, V, O in SVOs:
            self.assertEqual(valid, RulesCheck(S, V, O, sent), (S, V, O))

    def test_to_V(self):
        sent = "They consider making a cake for their mother."
        SVOs = [
            [True, "They", "make", "a cake"],
            [True, "They", "consider", "their mother"],
        ]
        for valid, S, V, O in SVOs:
            self.assertEqual(valid, RulesCheck(S, V, O, sent), (S, V, O))

    def test_to_V(self):
        sent = "They planned to make a cake for their mother."
        SVOs = [
            [True, "They", "make", "a cake"],
            [True, "They", "planned", "their mother"],
        ]
        for valid, S, V, O in SVOs:
            self.assertEqual(valid, RulesCheck(S, V, O, sent), (S, V, O))

    def test_wh_clause_1(self):
        sent = "The nation 's health maintenance organizations were required to tell the federal government by midnight Monday whether they plan to continue providing health insurance to Medicare recipients next year , raise premiums , or reduce benefits ."
        SVOs = [
            [True, "The nation 's health maintenance organizations",
                "tell", "federal government"],
            [True, "they", "providing", "insurance"],
        ]
        for valid, S, V, O in SVOs:
            self.assertEqual(valid, RulesCheck(S, V, O, sent), (S, V, O))

    def test_wh_clause_2(self):
        sent = "A Spanish official , who had just finished a siesta and seemed not the least bit tense , offered what he believed to be a perfectly reasonable explanation for why the portable facilities were n't in service ."
        SVOs = [
            [True, "who", "finished", "a siesta"],
            [True, "A Spanish official", "finished", "siesta"],
        ]
        for valid, S, V, O in SVOs:
            self.assertEqual(valid, RulesCheck(S, V, O, sent), (S, V, O))


# python -m unittest test
if __name__ == "__main__":
    unittest.main()
