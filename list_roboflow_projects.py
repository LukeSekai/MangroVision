"""
List all projects in your Roboflow workspace
"""
from roboflow import Roboflow

API_KEY = "LJti0618t62VdAV816QP"

print("\n" + "="*70)
print("🔍 Listing Your Roboflow Projects")
print("="*70 + "\n")

try:
    rf = Roboflow(api_key=API_KEY)
    
    # Get workspace
    print("📡 Connecting to Roboflow...")
    workspace = rf.workspace()
    
    print(f"✅ Workspace: {workspace.name}\n")
    print("📂 Your Projects:")
    print("-" * 70)
    
    # List all projects
    projects = workspace.projects()
    
    if not projects:
        print("   No projects found")
    else:
        for i, project in enumerate(projects, 1):
            print(f"\n{i}. Project Name: {project.name}")
            print(f"   Project ID: {project.id}")
            print(f"   Use in script: PROJECT_NAME = \"{project.id}\"")
    
    print("\n" + "="*70)
    print("✅ Done! Update your download script with the correct PROJECT_NAME")
    print("="*70 + "\n")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nTroubleshooting:")
    print("- Check your API key is correct")
    print("- Ensure you have internet connection")
