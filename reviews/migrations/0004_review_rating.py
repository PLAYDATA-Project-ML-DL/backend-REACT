# Generated by Django 4.1.7 on 2023-03-08 08:02

import django.core.validators
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('reviews', '0003_alter_review_medicine_alter_review_writer'),
    ]

    operations = [
        migrations.AddField(
            model_name='review',
            name='rating',
            field=models.PositiveSmallIntegerField(default=0, validators=[django.core.validators.MinLengthValidator(0), django.core.validators.MaxLengthValidator(5)]),
        ),
    ]
