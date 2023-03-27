from django.db import models
from common.models import CommonModel


""" 일반 장고 모델 """
class Medicine(models.Model):

    """ Model Definition for Medicines """
    class EtcChoices(models.TextChoices):
        ETC = "ETC", "전문의약품"
        OTC = "OTC", "일반의약품"

    permission_writer = models.ForeignKey(
        "users.User",
        on_delete=models.CASCADE,
        #on_delete = models.SET_NULL,
        null = True,
        related_name='Medicine',
    )

    name = models.TextField(
        verbose_name = "의약품명",
        )
    # 약 이름

    basis = models.TextField(
        verbose_name = "주 성분",
        )
    # 주 성분

    effect = models.TextField(
        verbose_name = "약효",
    )
    # 효과

    caution = models.TextField(
        verbose_name = "주의사항",
        null=True,
    )
    # 주의사항

    cautionOtherMedicines = models.TextField(
        verbose_name="함께 복용 불가능한 의약품 목록",
        null=True,
        )
    # 다른 약과 함께 복용 가능한지의 여부

    
    etcChoices = models.CharField(
        verbose_name="전문의약품(ETC), 일반의약품(OTC)",
        choices=EtcChoices.choices,
        max_length=3,
        blank=True,
    )

    #reviewList = 

    # 평균 평점을 출력해준다.
    def rating(self):
        count = self.reviews.count()
        if count == 0:
            return "No Reviews"
        else:
            total_rating = 0
            for review in self.reviews.all().values("rating"):
                total_rating += review["rating"]
            return round(total_rating / count, 2)
    
    # 연동된 review 수
    def reviews_count(self):
        count = self.reviews.count()
        if count == 0:
            return "No Reviews"
        else:
            return count

    def __str__(self) -> str:
        return self.name
    
class Comment(CommonModel):

    writer = models.ForeignKey(
        "users.User",
        on_delete=models.CASCADE,
        #on_delete = models.SET_NULL,
        null = True,
        related_name='comment',
    )

    medicine = models.ForeignKey(
        "medicines.Medicine",
        on_delete=models.CASCADE,
        #on_delete = models.SET_NULL,
        null = True,
        related_name='comment',
    )

    content = models.TextField(
        null=False,
        blank=False,
    )

    def __str__(self) -> str:
        return self.content
    








