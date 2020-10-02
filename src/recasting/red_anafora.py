#!/usr/bin/env python3.7
#coding=utf-8
'''
Reader for RED code

@author: Tim O'Gorman (timjogorman@gmail.com)
@since: 2020-04-01
'''
import logging
import bs4
import argparse
from pathlib import Path
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger('redparser')

# logger.setLevel(logging.INFO)

class RedMention:
    def __init__(self, eid, spanbox, etype, parentstype, features):
        self.eid = eid
        self.spanbox = spanbox
        self.entitytype = etype
        self.pt = parentstype
        self.feature_box = features
        self.mention_strings = []
    def add_strings(self, raw_text):
        for each_span_pair in self.spanbox:
            its_string = raw_text[each_span_pair[0]:each_span_pair[1]+1].replace("\n"," ")
            self.mention_strings.append(its_string)
    def __str__(self):
        return f"{' '.join(self.mention_strings)}({self.entitytype})"

    @classmethod
    def fromAnafora(cls, anafora_xml):
        eid = anafora_xml.find("id").text
        all_spans = [[int(y) for y in each_span.split(",")] for each_span in anafora_xml.find("span").text.split(";")]
        etype = anafora_xml.find("type").text
        ptype = anafora_xml.find("parentstype").text
        feature_box = {}
        for q in anafora_xml.find("properties").descendants:
            try:
                feature_box[q.name] = q.text
            except:
                pass
        if etype == 'EVENT':
            return Event(eid, all_spans, etype, ptype, feature_box)
        elif etype == 'ENTITY':
            return Entity(eid, all_spans, etype, ptype, feature_box)
        elif etype in ['TIMEX3','DOCTIME','SECTIONTIME']:
            return TempEx(eid, all_spans, etype, ptype, feature_box)
        
        else:
            print(etype)
        return RedMention(eid, all_spans, etype, ptype, feature_box)

class Event(RedMention):
    def __init__(self, eid, spanbox, etype, parentstype, features):
        self.eid = eid
        self.spanbox = spanbox
        self.entitytype = etype
        self.pt = parentstype
        self.feature_box = features
        self.modality = features['contextualmodality']
        self.doctimerel = features['doctimerel']
        self.aspecttype = features['type']
        self.implicit = features['representation']
        self.degree = features['degree']
        self.polarity = features['polarity']
        self.aspect = features['contextualaspect']
        self.mention_strings = []

    def __str__(self):
        return f"{' '.join(self.mention_strings)}({self.entitytype}: {self.doctimerel} , {self.modality} , {self.polarity})"

class Entity(RedMention):
    def __init__(self, eid, spanbox, etype, parentstype, features):
        self.eid = eid
        self.spanbox = spanbox
        self.entitytype = etype
        self.pt = parentstype
        self.feature_box = features
        self.modality = features['contextualmodality']
        self.polarity = features['polarity']
        self.mention_strings = []

    def __str__(self):
        return f"{' '.join(self.mention_strings)}({self.entitytype})"

class TempEx(RedMention):
    def __init__(self, eid, spanbox, etype, parentstype, features):
        self.eid = eid
        self.spanbox = spanbox
        self.entitytype = etype
        self.pt = parentstype

        self.feature_box = features
        self.timetype = features.get('class',self.entitytype)
        self.mention_strings = []
        
    def __str__(self):
        return f"{' '.join(self.mention_strings)}({self.entitytype}: {self.timetype})"

class RedRelation:
    def __init__(self, rel_id, rtype, parentstype, edge_box, features):
        self.rel_id = rel_id
        self.relation_type = rtype
        self.pt = parentstype
        self.edge_box = edge_box
        self.feature_box = features
        #print(self.rel_id, self.feature_box, self.relation_type)
    @classmethod
    def fromAnafora(cls, anafora_xml):
        eid = anafora_xml.find("id").text
        rtype = anafora_xml.find("type").text
        ptype = anafora_xml.find("parentstype").text
        feature_box = {}
        edge_box = {}
        for q in anafora_xml.find("properties").descendants:

            try:
                if q.name in ["type", 'polarity','contextualmodality','difficulty']:
                    feature_box[q.name] = q.text
                else:
                    edge_box[q.name] =edge_box.get(q.name, []) + [q.text]
            except:
                pass
        return RedRelation(eid, rtype, ptype, edge_box, feature_box)

class RedFile:
    def __init__(self, filename, mention_dictionary , relation_dictionary, raw_text):
        """Object for a whole RED annotated document"""

        self.filename = filename
        self.mention_dictionary =mention_dictionary
        self.relation_dictionary = relation_dictionary
        self.raw_text = raw_text
    @classmethod
    def fromAnafora(cls, anafora_file, text_file):
        """Read the xml and raw text file and produce the RED file """
        
        raw_text = open(text_file).read()
        mention_dictionary = {}
        relation_dictionary = {}
        for entity in bs4.BeautifulSoup(open(anafora_file),'lxml').find_all("entity"):
            anf = RedMention.fromAnafora(entity)
            anf.add_strings(raw_text)
            mention_dictionary[anf.eid] = anf
        for relation in bs4.BeautifulSoup(open(anafora_file),'lxml').find_all("relation"):
            rel_obj = RedRelation.fromAnafora(relation)
            relation_dictionary[rel_obj.rel_id] = rel_obj
        return RedFile(Path(text_file).name, mention_dictionary, relation_dictionary, raw_text)

    def print_relation(self, relation_id):
        current_relation = self.relation_dictionary[relation_id]
        output_string = f"{current_relation.relation_type}:\n"
        for slot_in_relation in current_relation.edge_box:
            output_string += f"\t{slot_in_relation}:\n" 
            for mention in current_relation.edge_box[slot_in_relation]:
                output_string += f"\t\t{str(self.mention_dictionary[mention])}\n"
        return output_string

    def print_mention(self, mention_id):

        return str(self.mention_dictionary[mention_id])


    def __str__(self):
        output = self.raw_text
        output += "\n".join([rf.print_mention(mention_id) for mention_id in rf.mention_dictionary])
        output += "\n".join([rf.print_relation(relation_id) for relation_id in rf.relation_dictionary])
        return output
    
if __name__ == "__main__":
    command_line_reader = argparse.ArgumentParser()
    command_line_reader.add_argument("input", help="RED xml file (RED-Relations.gold)")
    command_line_reader.add_argument("text", help="Raw text file")
    commands =command_line_reader.parse_args()
    rf = RedFile.fromAnafora(commands.input, commands.text)
    print(str(rf))