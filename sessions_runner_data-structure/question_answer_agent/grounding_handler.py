import json
import re
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

@dataclass
class GroundingInfo:
    """Data class to hold grounding information"""
    web_search_queries: List[str]
    grounding_chunks: List[Dict[str, Any]]
    rendered_content: Optional[str]
    search_urls: List[str]

class GroundingMetadataHandler:
    """
    Handler class for processing and displaying Google Search grounding metadata
    according to Google's policy requirements.
    """
    
    def __init__(self):
        self.grounding_info: Optional[GroundingInfo] = None
    
    def extract_grounding_info(self, response_event) -> Optional[GroundingInfo]:
        """
        Extract grounding information from the response event
        """
        grounding_metadata = None
        
        # Try different ways to access grounding metadata
        if hasattr(response_event, 'grounding_metadata'):
            grounding_metadata = response_event.grounding_metadata
        elif hasattr(response_event, 'candidates') and response_event.candidates:
            if len(response_event.candidates) > 0:
                candidate = response_event.candidates[0]
                if hasattr(candidate, 'grounding_metadata'):
                    grounding_metadata = candidate.grounding_metadata
        
        if not grounding_metadata:
            return None
        
        # Extract web search queries
        web_search_queries = []
        if hasattr(grounding_metadata, 'web_search_queries') and grounding_metadata.web_search_queries:
            web_search_queries = list(grounding_metadata.web_search_queries)
        
        # Extract grounding chunks (sources)
        grounding_chunks = []
        search_urls = []
        if hasattr(grounding_metadata, 'grounding_chunks') and grounding_metadata.grounding_chunks:
            for chunk in grounding_metadata.grounding_chunks:
                chunk_info = {}
                if hasattr(chunk, 'web') and chunk.web:
                    chunk_info['title'] = getattr(chunk.web, 'title', 'Unknown Title')
                    chunk_info['uri'] = getattr(chunk.web, 'uri', '')
                    if chunk_info['uri']:
                        search_urls.append(chunk_info['uri'])
                grounding_chunks.append(chunk_info)
        
        # Extract rendered content
        rendered_content = None
        if hasattr(grounding_metadata, 'search_entry_point') and grounding_metadata.search_entry_point:
            if hasattr(grounding_metadata.search_entry_point, 'rendered_content'):
                rendered_content = grounding_metadata.search_entry_point.rendered_content
        
        self.grounding_info = GroundingInfo(
            web_search_queries=web_search_queries,
            grounding_chunks=grounding_chunks,
            rendered_content=rendered_content,
            search_urls=search_urls
        )
        
        return self.grounding_info
    
    def display_console_grounding_info(self, grounding_info: GroundingInfo):
        """
        Display grounding information in console format
        """
        print("\n" + "="*60)
        print("GOOGLE SEARCH GROUNDING METADATA (REQUIRED BY GOOGLE POLICY)")
        print("="*60)
        
        if grounding_info.web_search_queries:
            print(f"🔍 Search Queries Used:")
            for i, query in enumerate(grounding_info.web_search_queries, 1):
                print(f"   {i}. \"{query}\"")
        
        if grounding_info.grounding_chunks:
            print(f"\n📄 Sources Used ({len(grounding_info.grounding_chunks)} sources):")
            for i, chunk in enumerate(grounding_info.grounding_chunks, 1):
                print(f"   {i}. {chunk.get('title', 'Unknown Title')}")
                if chunk.get('uri'):
                    print(f"      URL: {chunk['uri']}")
        
        if grounding_info.rendered_content:
            print(f"\n🎨 Google Search Suggestions HTML:")
            print("   (This HTML must be displayed in your application)")
            print(f"   Length: {len(grounding_info.rendered_content)} characters")
            
            # Extract search suggestion text for display
            suggestions = self.extract_search_suggestions_text(grounding_info.rendered_content)
            if suggestions:
                print(f"   Search Suggestions: {', '.join(suggestions)}")
        
        print("="*60)
        print("END GROUNDING METADATA")
        print("="*60 + "\n")
    
    def extract_search_suggestions_text(self, html_content: str) -> List[str]:
        """
        Extract search suggestion text from HTML content for display purposes
        """
        suggestions = []
        # Simple regex to extract text from what looks like search suggestion elements
        # This is a basic implementation - you might need to adjust based on actual HTML structure
        text_pattern = r'>([^<]+)</button>'
        matches = re.findall(text_pattern, html_content)
        
        for match in matches:
            clean_text = match.strip()
            if clean_text and len(clean_text) > 2:  # Filter out very short matches
                suggestions.append(clean_text)
        
        return suggestions
    
    def save_grounding_html(self, grounding_info: GroundingInfo, filename: str = "google_search_suggestions.html"):
        """
        Save the rendered content to an HTML file for testing/viewing
        """
        if not grounding_info.rendered_content:
            print("No rendered content to save.")
            return False
        
        try:
            html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Google Search Suggestions - Required Display</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 800px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .header {{
            background-color: #1a73e8;
            color: white;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        .info {{
            background-color: #e8f0fe;
            padding: 15px;
            border-left: 4px solid #1a73e8;
            margin-bottom: 20px;
        }}
        .grounding-content {{
            border: 1px solid #dadce0;
            border-radius: 8px;
            padding: 15px;
            background-color: #fafafa;
        }}
        .queries {{
            margin-bottom: 15px;
        }}
        .sources {{
            margin-bottom: 15px;
        }}
        .source-item {{
            margin: 5px 0;
            padding: 8px;
            background: white;
            border-radius: 4px;
            border-left: 3px solid #34a853;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Google Search Suggestions</h1>
            <p>Required display according to Google's grounding policies</p>
        </div>
        
        <div class="info">
            <h3>Important Notice</h3>
            <p>This content must be displayed when using Google Search grounding according to Google's policies. 
            The search suggestions below must be shown exactly as provided.</p>
        </div>
        
        <div class="queries">
            <h3>Search Queries Used:</h3>
            <ul>
                {''.join([f'<li>"{query}"</li>' for query in grounding_info.web_search_queries])}
            </ul>
        </div>
        
        <div class="sources">
            <h3>Sources:</h3>
            {''.join([f'<div class="source-item"><strong>{chunk.get("title", "Unknown")}</strong><br><small>{chunk.get("uri", "")}</small></div>' for chunk in grounding_info.grounding_chunks])}
        </div>
        
        <div class="grounding-content">
            <h3>Google Search Suggestions (Required Display):</h3>
            {grounding_info.rendered_content}
        </div>
    </div>
    
    <script>
        // Ensure any search suggestion clicks go to Google SRP as required
        document.addEventListener('click', function(e) {{
            // This would handle the tap-to-SRP requirement
            // Actual implementation would depend on the structure of rendered_content
            console.log('Search suggestion clicked:', e.target);
        }});
    </script>
</body>
</html>
            """
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(html_template)
            
            print(f"✅ Google Search Suggestions saved to '{filename}'")
            print(f"   Open this file in a browser to see the required display format.")
            return True
            
        except Exception as e:
            print(f"❌ Error saving HTML file: {e}")
            return False
    
    def get_compliance_checklist(self) -> Dict[str, bool]:
        """
        Check compliance with Google's requirements
        """
        if not self.grounding_info:
            return {"has_grounding_data": False}
        
        checklist = {
            "has_grounding_data": True,
            "has_web_search_queries": bool(self.grounding_info.web_search_queries),
            "has_grounding_chunks": bool(self.grounding_info.grounding_chunks),
            "has_rendered_content": bool(self.grounding_info.rendered_content),
            "has_search_urls": bool(self.grounding_info.search_urls)
        }
        
        checklist["fully_compliant"] = all([
            checklist["has_web_search_queries"],
            checklist["has_rendered_content"]
        ])
        
        return checklist
    
    def print_compliance_status(self):
        """
        Print compliance status with Google's requirements
        """
        checklist = self.get_compliance_checklist()
        
        print("\n📋 GOOGLE POLICY COMPLIANCE CHECK:")
        print("-" * 40)
        
        status_icon = "✅" if checklist.get("fully_compliant", False) else "⚠️"
        print(f"{status_icon} Overall Compliance: {'PASS' if checklist.get('fully_compliant', False) else 'NEEDS ATTENTION'}")
        
        for requirement, status in checklist.items():
            if requirement != "fully_compliant":
                icon = "✅" if status else "❌"
                print(f"{icon} {requirement.replace('_', ' ').title()}: {'Yes' if status else 'No'}")
        
        if not checklist.get("fully_compliant", False):
            print("\n⚠️  To comply with Google's policies, ensure:")
            print("   1. Search queries are displayed")
            print("   2. Rendered content (Search Suggestions) is shown")
            print("   3. Search suggestions link directly to Google SRP")
        
        print("-" * 40 + "\n")