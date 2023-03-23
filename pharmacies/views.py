from rest_framework.views import APIView
from .models import Pharmacy
from .serializers import PharmacySerializer, PharmacyDetailSerializer
from rest_framework.response import Response
from rest_framework.exceptions import NotFound, NotAuthenticated


""" 엘라스틱 서치 테스트 """
from elasticsearch_dsl import Search
from elasticsearch.exceptions import NotFoundError
#from elasticsearch import Elasticsearch
from .serializers import PharmacyElasticSearchSerializer
from elasticsearch.helpers import bulk
from elasticsearch_dsl.query import Q





class ElasticSearch(APIView):
    # 엘라스틱 서치에 저장된 데이터 출력
    def get(self, request):
        # 검색어
        #client = ElasticSearch()using=client,
        query = request.GET.get('search','')
        s = Search().using('default').index('my_index')
        q = Q('multi_match', query=query, fields=['title^3','callNumber', 'address', 'coordinate_X', 'coordinate_Y'])
        # elasticsearch-dsl 패키지의 search 클래스를 사용하여
        # Elasticsearch에서 데이터를 검색하는 예시,
        # request에서 search 파라미터를 받아와서 Search 클래스로 엘라스틱서치 쿼리를 작성.
        try:
            response = s.query(q).execute()
            # excute() : 작성된 엘라스틱 서치 쿼리로 엘라스틱서치에 요청하여 결과를 받아옴.
            hits = response.hits.hits
            results = [hit.to_dict() for hit in hits]
            serialized_results = PharmacyElasticSearchSerializer(results, many=True).data
        except NotFoundError:
            serialized_results = []
        return Response({'results': serialized_results})
    

class Pharmacies(APIView):

    def get(self, request):
        try:
            page = request.query_params.get('page', 1)
            page = int(page)
        except ValueError:
            page = 1
        page_size = 10
        start = (page-1) * page_size
        end = start + page_size
        all_Pharmacies = Pharmacy.objects.all()[start:end]
        serializer = PharmacySerializer(all_Pharmacies, many=True)
        return Response(serializer.data)#Response(serializer.data)
    
    def post(self, request):
        serializer = PharmacySerializer(data=request.data)
        if serializer.is_valid():
            new_pharmacies = serializer.save()
            return Response(PharmacySerializer(new_pharmacies).data)
        else:
            return Response(serializer.errors)
        
class PharmacyDetail(APIView):
    def get_object(self, pk):
        try:
            return Pharmacy.objects.get(pk=pk)
        except Pharmacy.DoesNotExist:
            raise NotFound

    def get(self, request, pk):
        serializer = PharmacyDetailSerializer(self.get_object(pk))
        return Response(serializer.data)
    
    def put(self, request, pk):
        pass

    def delete(self, request, pk):
        pass

"""

from .models import Pharmacy
import csv
data = None
file_dir = 'D:/db/'

def read_data(table_name):
    with open(file_dir + f'{table_name}.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        global data
        data = list(reader)
    return

def footer(table_name, class_name, bulk_list):
    class_name.objects.bulk_create(bulk_list)

    with open(file_dir + f'{table_name}.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
    return

def add_pharmacies(request):
    read_data('pharmacies')
    if not data:
        return HttpResponse('Nothing to update')
    
    arr=[]
    for row in data:
        arr.append(Pharmacy())

    footer('pharmecies', Pharmacy, arr)
    return HttpResponse('Pharmacies table updated')
    """