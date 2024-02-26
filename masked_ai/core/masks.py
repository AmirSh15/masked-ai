"""
"""
import re
from abc import ABC, abstractmethod
from typing import Any, List

from english_words import get_english_words_set

import nltk

nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)

__allowed_names__ = [
    "NetAct",
    "cell",
    "support",
    "backup",
    "restore",
    "case",
    "Java",
    "basic",
    "issue",
    "Once",
    "feature",
    "enabled",
    "helpful",
    "failure",
    "set",
    "resolved",
    "button",
    "monitor",
    "follow",
    "up",
    "PM",
    "CM",
    "FM",
    "need",
    "node",
    "startup",
    "dump",
    "site",
    "check",
    "checked",
    "serial",
    "sleep",
    "silent",
    "config",
    "change",
    "changes",
    "normal",
    "operations",
    "operation",
    "upgrade",
    "traffic",
    "title",
    "control",
    "network",
    "plane",
    "snapshot",
    "installation",
    "unknown",
    "release",
    "software",
    "transport",
    "alarm",
    "alarms",
    "time",
    "lte",
    "attempts",
    "setup",
    "degraded",
    "detection",
    "sleeping",
    "due",
    "reset",
    "recovery",
    "manual",
    "manager",
    "download",
    "suspect",
    "clear",
    "power",
    "setting",
    "confirm",
    "fix",
    "correction",
    "top",
    "reference",
    "capture",
]
__allowed_names__ = list(get_english_words_set(['web2'], lower=True))
__extensions__ =[
    "exe",
    "yml",
    "com",
    "sh",
]
__punctuation__ = ".,;:'"


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
    
## skip for the time being
# class NETParMask(MaskBase):
#     """Network parameters
#     """
#     @staticmethod
#     def find(data: str) -> List[Any]:
#         candidates = re.findall(r"(\w+(?:\.\w+)+)", data)
#         # make sure the last part is not an extension
#         new_candidates = []
#         for x in candidates:
#             if any([x.split('.')[-1].lower() in k.lower() for k in __extensions__]):
#                 continue
#             new_candidates.append(x)
        
#         new_candidates = list(set(new_candidates))
        
#         return new_candidates


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
            
        # if there is a picked word that has multiple names in it, add each individual to the person_list
        new_person_list = person_list
        for name in person_list: 
            start, end = data.find(name), data.find(name) + len(name)   
            if len(name.split(' ')) > 1 and data[start-1] != '''"''' and data[end] != '''"''':
                new_person_list.extend(name.split(' '))
        person_list = new_person_list
            
        # make sure the person's name is a single word and doesn not have " in the begining of end of it
        new_person_list = []
        for name in person_list:
            start, end = data.find(name), data.find(name) + len(name)
            # make sure the entity is not in the allowed_names. also avoid names here with all capital letters
            if len(name.split(' ') ) > 1 or \
                data[start-1] == '''"''' or data[end] == '''"''' or \
                any([name.lower() == x.lower() for x in __allowed_names__]) or \
                data[start-1] != ' ' or \
                (data[end] != ' ' and data[end] not in __punctuation__) or \
                name.isupper() or \
                len(name) == 1:
                continue
            new_person_list.append(name)
            
        new_person_list = list(set(new_person_list))
        
        return new_person_list


class LinkMask(MaskBase):
    """Web links
    """
    @staticmethod
    def find(data: str) -> List[Any]:
        return re.findall(r'(?i)((?:https?://|www\d{0,3}[.])?[a-z0-9.\-]+[.](?:(?:international)|(?:construction)|(?:contractors)|(?:enterprises)|(?:photography)|(?:immobilien)|(?:management)|(?:technology)|(?:directory)|(?:education)|(?:equipment)|(?:institute)|(?:marketing)|(?:solutions)|(?:builders)|(?:clothing)|(?:computer)|(?:democrat)|(?:diamonds)|(?:graphics)|(?:holdings)|(?:lighting)|(?:plumbing)|(?:training)|(?:ventures)|(?:academy)|(?:careers)|(?:company)|(?:domains)|(?:florist)|(?:gallery)|(?:guitars)|(?:holiday)|(?:kitchen)|(?:recipes)|(?:shiksha)|(?:singles)|(?:support)|(?:systems)|(?:agency)|(?:berlin)|(?:camera)|(?:center)|(?:coffee)|(?:estate)|(?:kaufen)|(?:luxury)|(?:monash)|(?:museum)|(?:photos)|(?:repair)|(?:social)|(?:tattoo)|(?:travel)|(?:viajes)|(?:voyage)|(?:build)|(?:cheap)|(?:codes)|(?:dance)|(?:email)|(?:glass)|(?:house)|(?:ninja)|(?:photo)|(?:shoes)|(?:solar)|(?:today)|(?:aero)|(?:arpa)|(?:asia)|(?:bike)|(?:buzz)|(?:camp)|(?:club)|(?:coop)|(?:farm)|(?:gift)|(?:guru)|(?:info)|(?:jobs)|(?:kiwi)|(?:land)|(?:limo)|(?:link)|(?:menu)|(?:mobi)|(?:moda)|(?:name)|(?:pics)|(?:pink)|(?:post)|(?:rich)|(?:ruhr)|(?:sexy)|(?:tips)|(?:wang)|(?:wien)|(?:zone)|(?:biz)|(?:cab)|(?:cat)|(?:ceo)|(?:com)|(?:edu)|(?:gov)|(?:int)|(?:mil)|(?:net)|(?:onl)|(?:org)|(?:pro)|(?:red)|(?:tel)|(?:uno)|(?:xxx)|(?:ac)|(?:ad)|(?:ae)|(?:af)|(?:ag)|(?:ai)|(?:al)|(?:am)|(?:an)|(?:ao)|(?:aq)|(?:ar)|(?:as)|(?:at)|(?:au)|(?:aw)|(?:ax)|(?:az)|(?:ba)|(?:bb)|(?:bd)|(?:be)|(?:bf)|(?:bg)|(?:bh)|(?:bi)|(?:bj)|(?:bm)|(?:bn)|(?:bo)|(?:br)|(?:bs)|(?:bt)|(?:bv)|(?:bw)|(?:by)|(?:bz)|(?:ca)|(?:cc)|(?:cd)|(?:cf)|(?:cg)|(?:ch)|(?:ci)|(?:ck)|(?:cl)|(?:cm)|(?:cn)|(?:co)|(?:cr)|(?:cu)|(?:cv)|(?:cw)|(?:cx)|(?:cy)|(?:cz)|(?:de)|(?:dj)|(?:dk)|(?:dm)|(?:do)|(?:dz)|(?:ec)|(?:ee)|(?:eg)|(?:er)|(?:es)|(?:et)|(?:eu)|(?:fi)|(?:fj)|(?:fk)|(?:fm)|(?:fo)|(?:fr)|(?:ga)|(?:gb)|(?:gd)|(?:ge)|(?:gf)|(?:gg)|(?:gh)|(?:gi)|(?:gl)|(?:gm)|(?:gn)|(?:gp)|(?:gq)|(?:gr)|(?:gs)|(?:gt)|(?:gu)|(?:gw)|(?:gy)|(?:hk)|(?:hm)|(?:hn)|(?:hr)|(?:ht)|(?:hu)|(?:id)|(?:ie)|(?:il)|(?:im)|(?:in)|(?:io)|(?:iq)|(?:ir)|(?:is)|(?:it)|(?:je)|(?:jm)|(?:jo)|(?:jp)|(?:ke)|(?:kg)|(?:kh)|(?:ki)|(?:km)|(?:kn)|(?:kp)|(?:kr)|(?:kw)|(?:ky)|(?:kz)|(?:la)|(?:lb)|(?:lc)|(?:li)|(?:lk)|(?:lr)|(?:ls)|(?:lt)|(?:lu)|(?:lv)|(?:ly)|(?:ma)|(?:mc)|(?:md)|(?:me)|(?:mg)|(?:mh)|(?:mk)|(?:ml)|(?:mm)|(?:mn)|(?:mo)|(?:mp)|(?:mq)|(?:mr)|(?:ms)|(?:mt)|(?:mu)|(?:mv)|(?:mw)|(?:mx)|(?:my)|(?:mz)|(?:na)|(?:nc)|(?:ne)|(?:nf)|(?:ng)|(?:ni)|(?:nl)|(?:no)|(?:np)|(?:nr)|(?:nu)|(?:nz)|(?:om)|(?:pa)|(?:pe)|(?:pf)|(?:pg)|(?:ph)|(?:pk)|(?:pl)|(?:pm)|(?:pn)|(?:pr)|(?:ps)|(?:pt)|(?:pw)|(?:py)|(?:qa)|(?:re)|(?:ro)|(?:rs)|(?:ru)|(?:rw)|(?:sa)|(?:sb)|(?:sc)|(?:sd)|(?:se)|(?:sg)|(?:sh)|(?:si)|(?:sj)|(?:sk)|(?:sl)|(?:sm)|(?:sn)|(?:so)|(?:sr)|(?:st)|(?:su)|(?:sv)|(?:sx)|(?:sy)|(?:sz)|(?:tc)|(?:td)|(?:tf)|(?:tg)|(?:th)|(?:tj)|(?:tk)|(?:tl)|(?:tm)|(?:tn)|(?:to)|(?:tp)|(?:tr)|(?:tt)|(?:tv)|(?:tw)|(?:tz)|(?:ua)|(?:ug)|(?:uk)|(?:us)|(?:uy)|(?:uz)|(?:va)|(?:vc)|(?:ve)|(?:vg)|(?:vi)|(?:vn)|(?:vu)|(?:wf)|(?:ws)|(?:ye)|(?:yt)|(?:za)|(?:zm)|(?:zw))(?:/[^\s()<>]+[^\s`!()\[\]{};:\'".,<>?\xab\xbb\u201c\u201d\u2018\u2019])?)', data)

class SerialNumMask(MaskBase):
    """Serial numbers (duplicate of PhoneMask)
    """
    @staticmethod
    def find(data: str) -> List[Any]:
        candidate = re.findall(r'''((?:(?<![\d-])(?:\+?\d{1,3}[-.\s*]?)?(?:\(?\d{3}\)?[-.\s*]?)?\d{3}[-.\s*]?\d{4}(?![\d-]))|(?:(?<![\d-])(?:(?:\(\+?\d{2}\))|(?:\+?\d{2}))\s*\d{2}\s*\d{3}\s*\d{4}(?![\d-])))''', data)
        # make sure that before and after there is a space
        new_candidate = []
        for x in candidate:
            start, end = data.find(x), data.find(x) + len(x)
            if data[start-1] != ' ' or data[end] != ' ':
                continue
            new_candidate.append(x)
        return new_candidate

class PhoneMask(MaskBase):
    """Phone numbers
    """
    @staticmethod
    def find(data: str) -> List[Any]:
        candidate = re.findall(r'''((?:(?<![\d-])(?:\+?\d{1,3}[-.\s*]?)?(?:\(?\d{3}\)?[-.\s*]?)?\d{3}[-.\s*]?\d{4}(?![\d-]))|(?:(?<![\d-])(?:(?:\(\+?\d{2}\))|(?:\+?\d{2}))\s*\d{2}\s*\d{3}\s*\d{4}(?![\d-])))''', data)
        # make sure that before and after there is a space
        new_candidate = []
        for x in candidate:
            start, end = data.find(x), data.find(x) + len(x)
            if data[start-1] != ' ' or data[end] != ' ':
                continue
            new_candidate.append(x)
        return new_candidate


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
    def __init__(self, model='dslim/distilbert-NER', min_score=0.5):
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
            # make sure the entity is not in the allowed_names
            if data[start-1] != ' ' or \
                (data[end] != ' ' and data[end] not in __punctuation__) or \
                len(entity) == 1 or \
                '#' in entity or \
                any([entity.lower() == x.lower() for x in __allowed_names__]):
                continue
            new_selected_entities.append((label, entity))
            
        # remove duplicates
        new_selected_entities = list(set(new_selected_entities))
                
        return new_selected_entities
