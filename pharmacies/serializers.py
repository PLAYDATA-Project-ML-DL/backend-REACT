from rest_framework import serializers
from .models import Pharmacy
from elasticsearch_dsl import serializer

""" 어떻게 json형태로 표현할것인지 정의 """

# 수동 테스트 코드
class PharmacySerializer(serializers.ModelSerializer):
    class Meta:
        model = Pharmacy
        # Meta클래스를 선언한 후 model 변수에 Pharmacy 넣어주면 자동으로 Review의 모델을 읽어서 Serializer를 구성해준다.
        fields = (
            "title",
            "callNumber",
        )
        # 표기할 데이터를 정한다.
        # exclude = (), 제외할 항목 선택
        # fields = (), 표기할 항목 선택


class PharmacyDetailSerializer(serializers.ModelSerializer):
    class Meta:
        model = Pharmacy
        fields = "__all__"
        # 표기할 데이터를 정한다.
        
        # exclude = (), 제외할 항목 선택
        # fields = (), 표기할 항목 선택


class PharmacyElasticSearchSerializer(serializer.JSONSerializer, serializers.ModelSerializer):
    # ElasticSearch to Json 출력용 Serializer
    class Meta:
        model = Pharmacy
        fields = (
            "title",
            "callNumber",
            "address",
            "coordinate_X",
            "coordinate_Y",
        )