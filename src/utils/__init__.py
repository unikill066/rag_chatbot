# imports

import os
from pathlib import Path
from typing import List, Optional, Any, Type
from langchain_community.document_loaders import CSVLoader, UnstructuredExcelLoader, JSONLoader, PyPDFLoader, Docx2txtLoader, UnstructuredXMLLoader, UnstructuredMarkdownLoader
from langchain_core.documents import Document
from constants import SUPPORTED_EXTENSIONS