import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_mcp():
    print("Testing the ISL Master Server MCP endpoint...")
    
    server_params = StdioServerParameters(
        command="python",
        args=["../mcp_servers/isl_telehealth_mcp.py"],
        env=None
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # List tools
            tools = await session.list_tools()
            print("\nAvailable Tools:")
            for tool in tools.tools:
                print(f"- {tool.name}")
                
            # Call tool
            print("\nCalling simulate_patient_consultation with ['headache', 'severe']...")
            result = await session.call_tool("simulate_patient_consultation", arguments={"sign_language_tokens": ["headache", "three_days"]})
            print("\nResult Data:")
            print(result.content[0].text)

if __name__ == "__main__":
    asyncio.run(test_mcp())
