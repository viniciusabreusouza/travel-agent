from textwrap import dedent
from crewai import Crew

from travel_agent.agents.trip_agents import TripAgents
from travel_agent.settings import Settings
from travel_agent.tasks.trip_tasks import TripTasks

import os

settings = Settings()

openai_api_key = settings.OPENAI_API_KEY
os.environ["OPENAI_MODEL_NAME"] = 'gpt-4o'
os.environ["SERPER_API_KEY"] = settings.SERPER_API_KEY

class TravelAgency:
    def __init__(self, origin, destination, date_range, interests, visited_cities, preferred_currency):
        self.destination = destination
        self.origin = origin
        self.interests = interests
        self.date_range = date_range
        self.visited_cities = visited_cities
        self.preferred_currency = preferred_currency

    def run(self):
        agents = TripAgents()
        tasks = TripTasks()

        city_selector_agent = agents.city_selection_agent()
        local_expert_agent = agents.local_expert()
        travel_concierge_agent = agents.travel_concierge()


        identify_task = tasks.identify_task(
            city_selector_agent,
            self.origin,
            self.destination,
            self.interests,
            self.date_range,
            visited_cities,
        )
        gather_task = tasks.gather_task(
            local_expert_agent,
            self.origin,
            self.interests,
            self.date_range,
        )
        plan_task = tasks.plan_task(
            travel_concierge_agent, 
            self.origin,
            self.interests,
            self.date_range,
            self.preferred_currency
        )

        crew = Crew(
            agents=[
                city_selector_agent, local_expert_agent, travel_concierge_agent
            ],
            tasks=[identify_task, gather_task, plan_task],
            verbose=True
        )

        result = crew.kickoff()
        return result

if __name__ == "__main__":
    print('----- Welcome to the AgencyTravelAI -----')

    location_start = input(
        dedent("""
            From where will you be traveling from?
        """)
    )

    destination = input(
        dedent(f"""
            Do you have any city or country that you are interested to travel? 
            (Insert the city name or press NO in case you don't have)
        """)
    )

    date_range = input(
        dedent("""
            What is the date range you are interested in traveling?
        """)
    )

    interests = input(
        dedent("""
            What are some of your high level interests and hobbies?
        """)
    )

    visited_cities = input(
        dedent("""
            What countries that you have visited previously?
        """)
    )

    preferred_currency = input(
        dedent("""
            Which currency you would like to receive the result?
        """)
    )

    # preferred_destination = input(
    #     dedent("""
    #         preferred destination travel (beach, mountain, etc)?
    #     """)
    # )

    

    travel_agency = TravelAgency(
        location_start, 
        destination, 
        date_range, 
        interests, 
        visited_cities,
        preferred_currency
    )
    result = travel_agency.run()
    print("\n\n########################")
    print("## Here is you Trip Plan")
    print("########################\n")
    print(result)