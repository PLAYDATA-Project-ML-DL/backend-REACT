from .models import Review
# 모든 Review의 정보를 가져와서 출력하기 위해
from .serializers import ReviewListSerializer, ReviewDetailSerializer
# 커스텀 번역기 코드 import
from rest_framework.viewsets import ModelViewSet
# APIView 대체..
from rest_framework.decorators import api_view
# rest_framework를 사용하기 위한 데코레이터 세팅
from rest_framework.response import Response
from rest_framework.status import HTTP_204_NO_CONTENT
from rest_framework.views import APIView
from rest_framework.exceptions import NotFound, NotAuthenticated, PermissionDenied
from django.shortcuts import render
"""
class ReviewViewSet(ModelViewSet):
    serializer_class = ReviewListSerializer
    queryset = Review.objects.all()
"""


def hello_word(request):
    return render(request, 'base.html')
#APIView를 활용한 데이터 관리 코드.

class Reviews(APIView):
    #template_name = 'reviews/contact.html'

    def get(self, request):
        try:
            page = request.query_params.get('page', 1)
            page = int(page)
        except ValueError:
            page = 1
        page_size = 10
        start = (page-1) * page_size
        end = start + page_size
        all_reviews = Review.objects.all()[start:end]
        serializer = ReviewListSerializer(all_reviews, many=True)
        return Response(serializer.data)

    def post(self, request):
        if request.user.is_authenticated:
            print(dir(request.user))
            # 사용자 인증
            # 또한 serializer가 save를 실행하기 전에 writer의 정보를 serializer를 통해서 전달해야한다.
            # request에서 받아온 user정보를 활용하면 현재 로그인한 유저의 정보를 가져오던가, 로그인하지 않았다면 익명의 정보를 가져올것이다.
            serializer = ReviewDetailSerializer(data=request.data)
            # user가 제공하는 데이터를 가지고 오고 싶다면 받은 data를 넘겨야한다.
            # ReviewSerializser는 보안을 위해 일부 정보만 가질 수 있지만 
            # ReviewDetailSerializer는 __all__선언으로 인해 모든 데이터를 받을 수 있다.
            # 이로 인해 유효성 검사를 수행할 수 있다, 다만 기본적으로 선언된 serializer의 데이터 구조를 충족시켜야한다.

            if serializer.is_valid():
                new_review = serializer.save(writer=request.user)
                # 새로운 리뷰 생성시 로그인한 유저의 정보를 자동으로 불러와서 사용한다.
                # writer=request.user가 validated_data에 자동으로 추가된다.(serializer의 create함수의 부분에)
                serializer = ReviewDetailSerializer(new_review)
                # 유저에게서 받은 데이터를 활용하여 Review 인스턴스를 생성해준다.
                # serializers.py에서 class를 생성해주면 해당 부분에서 처리를 진행한다.
                return Response(serializer.data)
            else:
                return Response(serializer.errors)
        else:
            raise NotAuthenticated

class ReviewDetail(APIView):
    

    def get_object(self, pk):
        try:
            return Review.objects.get(pk=pk)
        except Review.DoesNotExist:
            raise NotFound
    # django REST API의 컨벤션, API의 상세한 부분을 찾을 때는 해당 함수를 거쳐서 진행되게끔 작성하는게 좋다.

    def get(self, request, pk):
        serializer = ReviewDetailSerializer(self.get_object(pk))
        return Response(serializer.data)

    def put(self, request, pk):
        review = self.get_object(pk)
        if not request.user.is_authenticated:
            raise NotAuthenticated
        if request.user != request.writer:
            raise PermissionDenied
        serializer = ReviewDetailSerializer(
            self.get_object(pk),
            data=request.data,
            partial=True,
            )
        if serializer.is_valid():
            updated_review = serializer.save(writer=request.user)
            return Response(ReviewDetailSerializer(updated_review).data)
        else:
            return Response(serializer.errors)

    def delete(self, request, pk):
        review = self.get_object(pk)
        # 1. 유저가 아니면 삭제할 수 없다.
        if not request.user.is_authenticated:
            raise NotAuthenticated
        # 2. 작성자가 아니면 삭제할 수 없다.
        if request.user != request.writer:
            raise PermissionDenied
        review.delete()
        return Response(status=HTTP_204_NO_CONTENT)

    # from django.http import JsonResponse
    # 일반 스트링이 아닌 JsonResponse를 return해야 한다.
    # from django.core import serializers
    # QuerySet을 변환시켜주는 framework 

