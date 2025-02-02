import sys
import click
import signal
from rich.console import Console
from kb_client.knowledge_base_api_client import Client
from kb_client.knowledge_base_api_client.models import QueryRequest
from kb_client.knowledge_base_api_client.api.default import query_query_post

console = Console()

def handle_sigint(signum, frame):
    """Handles the SIGINT signal (Ctrl-C) to exit gracefully."""
    click.echo('\nExiting...')
    exit(0)

@click.command()
def cli():
    signal.signal(signal.SIGINT, handle_sigint)

    client = Client(base_url="http://localhost:8000")
    
    while True:
        try:
            question = click.prompt("\nPrompt", type=str)
            if not question:
                break
                

            request = QueryRequest(question=question)
            response = query_query_post.sync(client=client, body=request)
            
            console.print(f"\nAnswer: {response.answer}")
            console.print("\nSources:")
            for source in response.sources:
                console.print(f"- {source.title} ({source.source})")
                
        except (KeyboardInterrupt, EOFError):
            console.print("\nExiting...")
            sys.exit(0)
        except Exception as e:
            console.print(f"Error: {e}", style="red")

if __name__ == "__main__":
    cli()
