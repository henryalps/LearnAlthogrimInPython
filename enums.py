# -*- coding: utf-8 -*-

class Types:
    typeNames = []

    @staticmethod
    def get_type_name(type_value):
        return Types.typeNames[type_value]

    @staticmethod
    def get_type_num(obj):
        return len(obj.typeNames)

class BPTypes(Types):
    SBP, DBP = range(2)
    typeNames = ['SBP', 'DBP']

    @staticmethod
    def get_type_name(type_value):
        return BPTypes.typeNames[type_value]


class BHSTypes(Types):
    TYPE_A, TYPE_B, TYPE_C, TYPE_ERROR = range(4)
    typeNames = ['TYPE_A', 'TYPE_B', 'TYPE_C', 'TYPE_ERROR']

    @staticmethod
    def get_type_name(type_value):
        return BHSTypes.typeNames[type_value]
