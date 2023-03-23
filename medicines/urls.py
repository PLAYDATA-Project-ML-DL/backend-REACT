from django.urls import path
from . import views

urlpatterns = [
    path("", views.Medicines.as_view()),
    path("<int:pk>", views.MedicineDetail.as_view()),
    path("<int:pk>/reviews", views.MedicineReview.as_view()),
    path("<int:pk>/comments", views.Comments.as_view()),
    #path('defaultsearch/',views.searchMedicine, name="searchmedicine"),
    path('ocrsearch/',views.searchOcrResult.as_view()),
    path('search_result/',views.searchMedicineResult.as_view()),
    #path('elasticsearch/',views.ElasticSearch.as_view(), name='elasticsearch-search'),
    #path('elasticsave/',views.SaveToElasticsearchAPIView.as_view()),
]

# path('api/v1/medicines/', include("medicines.urls")),