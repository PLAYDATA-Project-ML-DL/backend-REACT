from django.urls import path
from . import views



urlpatterns = [
    path("test", views.hello_word),
    path("", views.Reviews.as_view()),# path에 class를 가져오려면 .as_view()를 해줘야 한다.
    path("<int:pk>", views.ReviewDetail.as_view())# <자료형:파라미터 이름> : url에서 변수를 받을 수 있다.
    # path("comment/", view.Comments.as_view()) # 이런식으로 댓글확장 가능.
    ]
"""
# testwork API

urlpatterns = [
    path("", views.ReviewViewSet.as_view({
    'get':'list',
    'post':'create',
    })),# path에 class를 가져오려면 .as_view()를 해줘야 한다.
    path("<int:pk>", views.ReviewViewSet.as_view({
    'get':'retrieve',
    'put':'partial_update',
    'delete':'destroy',
    }))# <자료형:파라미터 이름> : url에서 변수를 받을 수 있다.
    ]
# 이미 config.urls.py를 타고 들어온 경로이기 때문에 ""부분은 "reviews/"와 같게 작동한다.
"""