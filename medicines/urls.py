from django.urls import path
from . import views

urlpatterns = [
    path("", views.Medicines.as_view()),
    path("<int:pk>", views.MedicineDetail.as_view()),
    path("<int:pk>/reviews", views.MedicineReview.as_view()),
    path("<int:pk>/comments", views.Comments.as_view()),
<<<<<<< HEAD
    #path('defaultsearch/',views.searchMedicine, name="searchmedicine"),
=======
    path('symptomsearch/',views.SearchSymptom.as_view()),
>>>>>>> 100bd06ea6d99ddb10bffcc0099e97928fb5d863
    path('ocrsearch/',views.searchOcrResult.as_view()),
    path('search_result/',views.searchMedicineResult.as_view()),

    path('search/', views.MedicineSearchAPIView.as_view(), name='medicine_search'),
    #path('elasticsave/',views.SaveToElasticsearchAPIView.as_view()),
]

# path('api/v1/medicines/', include("medicines.urls")),