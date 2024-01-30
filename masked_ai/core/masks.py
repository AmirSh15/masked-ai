"""
"""
import re
from abc import ABC, abstractmethod
from typing import Any, List

import nltk

nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)


class MaskBase(ABC):
    """Abstract class, how to implement new mask
    """

    @staticmethod
    @abstractmethod
    def find(data: str) -> List[Any]:
        """Implement this method

        :param data: Data to mask

        :return: New, masked data, and the loopup table to reconstruct it
        """
        return NotImplemented


class IPMask(MaskBase):
    """IP addresses
    """
    @staticmethod
    def find(data: str) -> List[Any]:
        return re.findall(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", data)


class NamesMask(MaskBase):
    """Persons names
    """
    @staticmethod
    def find(data: str) -> List[Any]:
        sentt = nltk.ne_chunk(nltk.pos_tag(nltk.tokenize.word_tokenize(data)), binary=False)
        person_list = []
        person = []
        name = ""
        for subtree in sentt.subtrees(filter=lambda t: t.label() == 'PERSON'):
            for leaf in subtree.leaves():
                person.append(leaf[0])
            for part in person:
                name += part + ' '
            if name[:-1] not in person_list:
                person_list.append(name[:-1])
            name = ''
            person = []
        return person_list


class LinkMask(MaskBase):
    """Web links
    """
    @staticmethod
    def find(data: str) -> List[Any]:
        return re.findall(r'(?i)((?:https?://|www\d{0,3}[.])?[a-z0-9.\-]+[.](?:(?:international)|(?:construction)|(?:contractors)|(?:enterprises)|(?:photography)|(?:immobilien)|(?:management)|(?:technology)|(?:directory)|(?:education)|(?:equipment)|(?:institute)|(?:marketing)|(?:solutions)|(?:builders)|(?:clothing)|(?:computer)|(?:democrat)|(?:diamonds)|(?:graphics)|(?:holdings)|(?:lighting)|(?:plumbing)|(?:training)|(?:ventures)|(?:academy)|(?:careers)|(?:company)|(?:domains)|(?:florist)|(?:gallery)|(?:guitars)|(?:holiday)|(?:kitchen)|(?:recipes)|(?:shiksha)|(?:singles)|(?:support)|(?:systems)|(?:agency)|(?:berlin)|(?:camera)|(?:center)|(?:coffee)|(?:estate)|(?:kaufen)|(?:luxury)|(?:monash)|(?:museum)|(?:photos)|(?:repair)|(?:social)|(?:tattoo)|(?:travel)|(?:viajes)|(?:voyage)|(?:build)|(?:cheap)|(?:codes)|(?:dance)|(?:email)|(?:glass)|(?:house)|(?:ninja)|(?:photo)|(?:shoes)|(?:solar)|(?:today)|(?:aero)|(?:arpa)|(?:asia)|(?:bike)|(?:buzz)|(?:camp)|(?:club)|(?:coop)|(?:farm)|(?:gift)|(?:guru)|(?:info)|(?:jobs)|(?:kiwi)|(?:land)|(?:limo)|(?:link)|(?:menu)|(?:mobi)|(?:moda)|(?:name)|(?:pics)|(?:pink)|(?:post)|(?:rich)|(?:ruhr)|(?:sexy)|(?:tips)|(?:wang)|(?:wien)|(?:zone)|(?:biz)|(?:cab)|(?:cat)|(?:ceo)|(?:com)|(?:edu)|(?:gov)|(?:int)|(?:mil)|(?:net)|(?:onl)|(?:org)|(?:pro)|(?:red)|(?:tel)|(?:uno)|(?:xxx)|(?:ac)|(?:ad)|(?:ae)|(?:af)|(?:ag)|(?:ai)|(?:al)|(?:am)|(?:an)|(?:ao)|(?:aq)|(?:ar)|(?:as)|(?:at)|(?:au)|(?:aw)|(?:ax)|(?:az)|(?:ba)|(?:bb)|(?:bd)|(?:be)|(?:bf)|(?:bg)|(?:bh)|(?:bi)|(?:bj)|(?:bm)|(?:bn)|(?:bo)|(?:br)|(?:bs)|(?:bt)|(?:bv)|(?:bw)|(?:by)|(?:bz)|(?:ca)|(?:cc)|(?:cd)|(?:cf)|(?:cg)|(?:ch)|(?:ci)|(?:ck)|(?:cl)|(?:cm)|(?:cn)|(?:co)|(?:cr)|(?:cu)|(?:cv)|(?:cw)|(?:cx)|(?:cy)|(?:cz)|(?:de)|(?:dj)|(?:dk)|(?:dm)|(?:do)|(?:dz)|(?:ec)|(?:ee)|(?:eg)|(?:er)|(?:es)|(?:et)|(?:eu)|(?:fi)|(?:fj)|(?:fk)|(?:fm)|(?:fo)|(?:fr)|(?:ga)|(?:gb)|(?:gd)|(?:ge)|(?:gf)|(?:gg)|(?:gh)|(?:gi)|(?:gl)|(?:gm)|(?:gn)|(?:gp)|(?:gq)|(?:gr)|(?:gs)|(?:gt)|(?:gu)|(?:gw)|(?:gy)|(?:hk)|(?:hm)|(?:hn)|(?:hr)|(?:ht)|(?:hu)|(?:id)|(?:ie)|(?:il)|(?:im)|(?:in)|(?:io)|(?:iq)|(?:ir)|(?:is)|(?:it)|(?:je)|(?:jm)|(?:jo)|(?:jp)|(?:ke)|(?:kg)|(?:kh)|(?:ki)|(?:km)|(?:kn)|(?:kp)|(?:kr)|(?:kw)|(?:ky)|(?:kz)|(?:la)|(?:lb)|(?:lc)|(?:li)|(?:lk)|(?:lr)|(?:ls)|(?:lt)|(?:lu)|(?:lv)|(?:ly)|(?:ma)|(?:mc)|(?:md)|(?:me)|(?:mg)|(?:mh)|(?:mk)|(?:ml)|(?:mm)|(?:mn)|(?:mo)|(?:mp)|(?:mq)|(?:mr)|(?:ms)|(?:mt)|(?:mu)|(?:mv)|(?:mw)|(?:mx)|(?:my)|(?:mz)|(?:na)|(?:nc)|(?:ne)|(?:nf)|(?:ng)|(?:ni)|(?:nl)|(?:no)|(?:np)|(?:nr)|(?:nu)|(?:nz)|(?:om)|(?:pa)|(?:pe)|(?:pf)|(?:pg)|(?:ph)|(?:pk)|(?:pl)|(?:pm)|(?:pn)|(?:pr)|(?:ps)|(?:pt)|(?:pw)|(?:py)|(?:qa)|(?:re)|(?:ro)|(?:rs)|(?:ru)|(?:rw)|(?:sa)|(?:sb)|(?:sc)|(?:sd)|(?:se)|(?:sg)|(?:sh)|(?:si)|(?:sj)|(?:sk)|(?:sl)|(?:sm)|(?:sn)|(?:so)|(?:sr)|(?:st)|(?:su)|(?:sv)|(?:sx)|(?:sy)|(?:sz)|(?:tc)|(?:td)|(?:tf)|(?:tg)|(?:th)|(?:tj)|(?:tk)|(?:tl)|(?:tm)|(?:tn)|(?:to)|(?:tp)|(?:tr)|(?:tt)|(?:tv)|(?:tw)|(?:tz)|(?:ua)|(?:ug)|(?:uk)|(?:us)|(?:uy)|(?:uz)|(?:va)|(?:vc)|(?:ve)|(?:vg)|(?:vi)|(?:vn)|(?:vu)|(?:wf)|(?:ws)|(?:ye)|(?:yt)|(?:za)|(?:zm)|(?:zw))(?:/[^\s()<>]+[^\s`!()\[\]{};:\'".,<>?\xab\xbb\u201c\u201d\u2018\u2019])?)', data)


class PhoneMask(MaskBase):
    """Phone numbers
    """
    @staticmethod
    def find(data: str) -> List[Any]:
        return re.findall(r'''((?:(?<![\d-])(?:\+?\d{1,3}[-.\s*]?)?(?:\(?\d{3}\)?[-.\s*]?)?\d{3}[-.\s*]?\d{4}(?![\d-]))|(?:(?<![\d-])(?:(?:\(\+?\d{2}\))|(?:\+?\d{2}))\s*\d{2}\s*\d{3}\s*\d{4}(?![\d-])))''', data)


class EmailMask(MaskBase):
    """Email addresses
    """
    @staticmethod
    def find(data: str) -> List[Any]:
        return re.findall(r"([a-z0-9!#$%&'*+\/=?^_`{|.}~-]+@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)", data)


class CreditCardMask(MaskBase):
    """Credit Card
    """
    @staticmethod
    def find(data: str) -> List[Any]:
        return re.findall(r'\b(?:\d{4}-){3}\d{4}|\b\d{16}\b', data)
    

class NERNamesMASK(MaskBase):
    """Named Entity Recognition using big NLP models
    
    """
    def __init__(self, model='dslim/distilbert-NER', min_score=0.4):
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        from transformers import pipeline

        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForTokenClassification.from_pretrained(model)
        
        self.nlp = pipeline("ner", model=self.model, tokenizer=self.tokenizer)
        
        self.__name__ = 'NERNamesMASK'
        self.min_score = min_score
        
        # mapping of NER tags to their descriptions
        if model == 'dslim/distilbert-NER':
            self.tag2name = {
                "LABEL_0": "O",       # Outside of a named entity
                # "LABEL_1": "B-MISC",  # Beginning of a miscellaneous entity right after another miscellaneous entity
                "LABEL_1": "MiscellMASK",  # Beginning of a miscellaneous entity right after another miscellaneous entity
                # "LABEL_2": "I-MISC",  # Miscellaneous entity
                "LABEL_2": "MiscellMASK",  # Miscellaneous entity
                # "LABEL_3": "B-PER",   # Beginning of a person's name right after another person's name
                "LABEL_3": "PersonMASK",   # Beginning of a person's name right after another person's name
                # "LABEL_4": "I-PER",   # Person's name
                "LABEL_4": "PersonMASK",   # Person's name
                # "LABEL_5": "B-ORG",   # Beginning of an organisation right after another organisation
                "LABEL_5": "OrgMASK",   # Beginning of an organisation right after another organisation
                # "LABEL_6": "I-ORG",   # Organisation
                "LABEL_6": "OrgMASK",   # Organisation
                # "LABEL_7": "B-LOC",   # Beginning of a location right after another location
                "LABEL_7": "LocationMASK",   # Beginning of a location right after another location
                # "LABEL_8": "I-LOC",    # Location
                "LABEL_8": "LocationMASK",    # Location
            }
        else:
            raise NotImplementedError(f"Mapping for model {model} not implemented!")
        
    def find(self, data: str) -> List[Any]:
        ner_results = self.nlp(data)
        # only return the entities
        selected_entities = [(entity['word'], self.tag2name[entity['entity']]) for entity in ner_results if (entity['entity'] == 'LABEL_3' or entity['entity'] == 'LABEL_5' or entity['entity'] == 'LABEL_6') and entity['score'] > self.min_score]
        
        # make sure the selected entities are a single word, not a subword --> check there is a space before and after
        new_selected_entities = []
        for (entity, label) in selected_entities:
            start, end = data.find(entity), data.find(entity) + len(entity)
            if data[start-1] != ' ' or data[end] != ' ' or '#' in entity:
                continue
            new_selected_entities.append((label, entity))
            
        # remove duplicates
        new_selected_entities = list(set(new_selected_entities))
                
        return new_selected_entities