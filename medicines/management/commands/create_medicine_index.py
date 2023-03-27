from django.core.management.base import BaseCommand
from elasticsearch import Elasticsearch
from medicines.models import Medicine

class Command(BaseCommand):
    help = 'Create Elasticsearch index for Medicine model'

    def handle(self, *args, **options):
        es = Elasticsearch()

        if es.indices.exists(index='medicine_index'):
            print('Index already exists')
            return

        es.indices.create(
            index='medicine_index',
            body={
                'settings': {
                    'number_of_shards': 1,
                    'number_of_replicas': 0,
                },
                'mappings': {
                    'properties': {
                        'name': {'type': 'text'},
                        'basis': {'type': 'text'},
                        'effect': {'type': 'text'},
                        'caution': {'type': 'text'},
                        'cautionOtherMedicines': {'type': 'text'},
                        'etcChoices': {'type': 'keyword'},
                    }
                }
            }
        )

        for medicine in Medicine.objects.all():
            es.index(
                index='medicine_index',
                id=medicine.id,
                body={
                    'name': medicine.name,
                    'basis': medicine.basis,
                    'effect': medicine.effect,
                    'caution': medicine.caution,
                    'cautionOtherMedicines': medicine.cautionOtherMedicines,
                    'etcChoices': medicine.etcChoices,
                }
            )

        print('Index created successfully')