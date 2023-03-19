from rest_framework.serializers import ModelSerializer
from rest_framework import serializers 
from .models import Medicine, Comment, MedicineElasticSearch
from reviews.serializers import ReviewListSerializer 
from elasticsearch_dsl import serializer



""" ElasticSearch Serializer """
class MedicineElasticSearchSerializer(serializer.JSONSerializer, ModelSerializer):
    """ ElasticSearch to Json 출력용 Serializer """
    class Meta:
        model = Medicine
        fields = (
            "pk",
            "name",
            "etcChoices",
            "rating",
        )

class MedicineElasticSaveSerializer(ModelSerializer):
    """MySQL to ElasticSearch 저장용 Serializer """
    name = serializers.CharField(max_length=200)
    etcChoices = serializers.CharField(max_length=3)
    rating = serializers.IntegerField()

    def create(self, validated_data):
        return MedicineElasticSearch(**validated_data)
    
    def update(self, instance, validated_data):
        instance.pk = validated_data.get('pk', instance.pk)
        instance.name = validated_data.get('name', instance.name)
        instance.etcChoices = validated_data.get('etcChoices', instance.etcChoices)
        instance.rating = validated_data.get('rating', instance.rating)
        return instance



""" Django Serializer """

class CommentSerializer(ModelSerializer):
    class Meta:
        model = Comment
        fields = (
            'medicine',
            'content',
            'created_at',
        )

class MedicineSerializer(ModelSerializer):
    class Meta:
        model = Medicine
        fields = (
            "pk",
            "name",
            "etcChoices",
            "rating",
        )


class MedicineDetailSerializer(ModelSerializer):
    #comment = CommentSerializer(read_only=True, many=True)
    # Medicine과 Comment는 relationship이기 때문에 수동으로 연결해줘야한다. 
    reviews = ReviewListSerializer(read_only=True, many=True)
    class Meta:
        model = Medicine
        fields = (
            "name",
            "basis",
            "effect",
            "caution",
            "cautionOtherMedicines",
            "etcChoices",
            "rating",
            "reviews_count",
            "reviews",
        )


class TinyMedicineSerializer(ModelSerializer):
    class Meta:
        model = Medicine
        fields = (
            'name',
            'etcChoices',
        )

