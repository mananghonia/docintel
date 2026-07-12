from django.contrib import admin

from documents.models import Correction, Extraction, InvoiceDocument

admin.site.register(InvoiceDocument)
admin.site.register(Extraction)
admin.site.register(Correction)
