# Generated by Django 4.1.7 on 2023-03-17 08:38

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('users', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='user',
            name='profile_photo',
            field=models.URLField(blank=True, null=True),
        ),
    ]