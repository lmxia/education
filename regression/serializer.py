from rest_framework import serializers

from linear_regression.models import Party

class PartySerializer(serializers.ModelSerializer):
    class Meta:
        model = Person
        fields = ('name', 'weight', 'height')