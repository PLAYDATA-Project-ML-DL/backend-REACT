from django.urls import path
from . import views


# testwork API

urlpatterns = [
    path("", views.Pharmacies.as_view()),
    path("<int:pk>", views.PharmacyDetail.as_view()),
    path("elasticsearch", views.ElasticSearch.as_view(), name='search'),
    ]
# 이미 config.urls.py를 타고 들어온 경로이기 때문에 ""부분은 "reviews/"와 같게 작동한다.
