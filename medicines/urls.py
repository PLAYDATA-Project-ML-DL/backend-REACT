from django.urls import path
from . import views

urlpatterns = [
    path("", views.Medicines.as_view()),
    path("<int:pk>", views.MedicineDetail.as_view()),
    path("<int:pk>/reviews", views.MedicineReview.as_view()),
    path("<int:pk>/comments", views.Comments.as_view()),
    path('symptomsearch/',views.SearchSymptom.as_view()),
    path('ocrsearch/',views.searchOcrResult.as_view()),
    path('search_result/',views.searchMedicineResult.as_view()),
    path('search_image/',views.MedicineImageSearch.as_view()),
    #엘라스틱 서치 패스"""
    path('search/', views.MedicineSearchAPIView.as_view(), name='medicine_search'),

    
    #path('elasticsave/',views.SaveToElasticsearchAPIView.as_view()),
]

# path('api/v1/medicines/', include("medicines.urls")),