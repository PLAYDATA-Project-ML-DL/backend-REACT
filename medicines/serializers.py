from rest_framework.serializers import ModelSerializer
from .models import Medicine, Comment
from reviews.serializers import ReviewListSerializer 

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
# SerializerMethodField를 사용하면 DB에 없는 내용이라도 출력할 수 있다.
# ex) ratingTest = serializers.SerializerMethodField()
# def get_ratingTest(self, medicine):
#     return meditine.updated_at
# 형식으로 호출하여 함수 이름으로 fields에 넣어줄 수 있다.

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
            #"is_admin",
        )
    #is_admin = serializers.SerializerMethodField()
    #def get_is_admin(self, medicine):
    #    request = self.context['request']
    #    return medicine.permission_writer == request.user



class TinyMedicineSerializer(ModelSerializer):
    class Meta:
        model = Medicine
        fields = (
            'name',
            'etcChoices',
        )

