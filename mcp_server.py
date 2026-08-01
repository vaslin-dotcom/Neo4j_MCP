from mcp.server.fastmcp import FastMCP
from pypdf import PdfReader
from docx import Document
from pathlib import Path

mcp=FastMCP('Neo4j-MCP')

@mcp.tool()
def reader(file_path:str)->str:
    """Read a PDF or DOCX file and return it in text format.

        Args:
            file_path: Absolute path to a .pdf or .docx file
        """
    path=Path(file_path)
    if not path.exists():
        return f'File {path.name} does not exist'

    elif path.suffix == '.docx':
        document=Document(str(path))
        text='\n'.join(p.text for p in document.paragraphs)
        return f'{file_path}\n\n{text}'

    elif path.suffix == '.pdf':
        reader = PdfReader(str(path))
        text=""
        for page in reader.pages:
            text+=page.extract_text() or ''
        return f'{file_path}\n\n{text}'
    else:
        return f'File {path.name} not supported'

if __name__=='__main__':
    mcp.run(transport='stdio')



