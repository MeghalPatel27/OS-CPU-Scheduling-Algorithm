import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit_shadcn_ui as ui
from matplotlib.animation import FuncAnimation
from streamlit_option_menu import option_menu


class Process:
    def __init__(self):
        self.pos = 0
        self.AT = 0
        self.BT = 0
        self.ST = []
        self.WT = 0
        self.FT = 0
        self.TAT = 0
def mru_page_replacement(process, memory_size):
    cache = []
    arr=[]
    page_faults = 0
    prev=None
    steps = []  # For visualizing the page replacement process step by step
    for page in process:
        
        if not page:
            continue  # Skip empty strings
        page = int(page)
        replaced_page = None  # Initialize replaced_page
        hit_or_miss=None
        if page not in cache:
            page_faults += 1
            if len(cache) == memory_size:
                replaced_page = int(prev)#int(cache[-1])  # Convert replaced page to integer
                #cache.pop()
                for i in range(len(cache)):
                  if cache[i]==prev:
                    cache[i]=page
                hit_or_miss="Miss"
            else:
                  cache.append(page)
                  hit_or_miss="Miss"
        # else:
            # cache.remove(page)
            # cache.append(page)
        else:
          hit_or_miss="Hit"
        prev=page
        
        
        steps.append((page, list(cache), replaced_page,hit_or_miss))  # Record the current cache state and replaced page after each step
    return page_faults, steps


def visualize_mru(steps):
    df = pd.DataFrame(steps, columns=["Page Accessed", "Cache State", "Page Replaced","Hit or Miss"])
    df['Page Replaced'] = df['Page Replaced'].apply(lambda x: int(x) if pd.notnull(x) else None)  # Convert replaced page to integer
    df['Page Replaced'] = df['Page Replaced'].astype('Int64')  # Convert to nullable integer type
    st.table(df)
    
def plot_gantt_chart(gantt_chart):
    fig = go.Figure()

    for i, row in enumerate(gantt_chart):
        start = row['Start']
        end = row['End']
        process = row['Process']
        
        # Skip adding trace if both start and end times are zero
        if start != 0 or end != 0:
            # Specify color only for selected processes
            color = f'rgb({i * 30 % 255}, {i * 50 % 255}, {i * 70 % 255})'
        else:
            # Set color to None for processes not selected
            color = None
            
        # Add a bar for each process from start to end time
        fig.add_trace(go.Bar(x=[start, end], y=[process, process],
                             orientation='h', name=f'Process {process}',
                             marker=dict(color=color)))
        
        # Add annotations to show the start and end times of each process
        fig.add_annotation(x=start, y=process, text=f'{start}', showarrow=False, font=dict(color='white', size=10))
        fig.add_annotation(x=end, y=process, text=f'{end}', showarrow=False, font=dict(color='white', size=10))

    fig.update_layout(title='Gantt Chart', xaxis_title='Time', yaxis_title='Process')
    st.plotly_chart(fig)


    
    
def mru_algorithm(process, memory_size):
    page_faults, steps = mru_page_replacement(process, memory_size)
    return page_faults, steps

def turn_around_time(arrival,finish):
  """
  Provide arrival time and finish time as array inputs returns 
  """
  turn_around_time=[]
  for x in range(len(arrival)):
    turn_around_time.append(finish[x]-arrival[x])
  return turn_around_time

def wait_time(turn_around,burst):
  """
  provide array input and will produce array as output
  """
  wait=[]
  for x in range(len(burst)):
    wait.append(turn_around[x]-burst[x])
  return wait

def make_dataframe(process,start,burst,finish,turn_around,wait):
  """
  provide the arrays for all attributes and converted dataframe will be returned
  """
  df=pd.DataFrame({"Process":process,'Arrival Time':start,"Burst Time":burst,'Completion Time':finish,'Turn Around Time':turn_around,'Waiting Time':wait}).sort_values(by ='Process' )
  return df







def is_safe(processes, avail, max_need, alloc):
    n = len(processes)
    m = len(avail)
    need = max_need - alloc
    finish = [False] * n
    safe_seq = []

    work = avail.copy()
    while True:
        found = False
        for i in range(n):
            if not finish[i]:
                if all(need[i] <= work):
                    work += alloc[i]
                    finish[i] = True
                    safe_seq.append(processes[i])
                    found = True

        if not found:
            break

    return all(finish), safe_seq

def banker_algorithm(processes, avail, max_need, alloc):
    st.write("Output:")
    df = pd.DataFrame(columns=["Process", "Resource Allocation", "Max Allocation", "Need"])

    need = max_need - alloc
    for i, process in enumerate(processes):
        df.loc[i] = [process, str(alloc[i]), str(max_need[i]), str(need[i])]

    st.write(df)

    safe, safe_seq = is_safe(processes, avail, max_need, alloc)
    if safe:
        st.write("Safe sequence found: <", " - ".join(safe_seq), ">")
    else:
        st.write("Unsafe state. No safe sequence found.")

    #st.write("Available resources after simulation:", avail)
    #st.write("Need matrix after simulation:")
    #st.write(max_need - alloc)

def scan_disk(requests, head, direction, max_track):
    # Sorting the requests
    requests.sort()
    total_tracks_moved = 0
    traversed_tracks = [head]
    minval=min(requests)
    maxval=max(requests)

    # Move right
    if direction == 'Right':
        right_requests = [track for track in requests if track >= head]
        
        left_requests = [track for track in requests if track < head]
        

        right_requests.append(max_track)
        right_requests.sort()

        traversed_tracks += right_requests
        
        # Traversing to the maximum track

        # Handling remaining tracks in decreasing order
        remaining_requests = [track for track in requests if track < head]
        
        remaining_requests.sort(reverse=True)
        traversed_tracks += remaining_requests

        total_tracks_moved = (max_track-head)+(max_track-minval)

    # Move left
    elif direction == 'Left':
        left_requests = [track for track in requests if track <= head]
        right_requests = [track for track in requests if track > head]

        left_requests.sort(reverse=True)
        left_requests.append(0)
        
        print("left",left_requests)

        traversed_tracks += left_requests
        #traversed_tracks.sort()
        print("trav",traversed_tracks)
        # Traversing to the minimum track

        # Handling remaining tracks in decreasing order
        remaining_requests = [track for track in requests if track > head]
        remaining_requests.sort()
        traversed_tracks += remaining_requests

        total_tracks_moved =(head)+(maxval)

    return traversed_tracks, total_tracks_moved


def app_layout():
  st.set_page_config(layout="wide")

   
  selected = option_menu(
        menu_title=None,
        options=["Round Robin", "Banker's Algorithm", "Scan-Disk Scheduling", "Most Recently Used [MRU] Page Repelacment"],
        orientation="horizontal",
    )
  algo=selected
  
  if "Banker's Algorithm" in algo:
        st.subheader("Banker's Algorithm")
        st.write("The Banker's algorithm is a deadlock avoidance algorithm used in operating systems to manage the allocation of resources to multiple processes in a way that prevents deadlock. It was developed by Dijkstra. The algorithm works by considering the maximum possible demand of each process and ensuring that the system remains in a safe state by granting resource requests only if the resulting state will still be safe.")
        st.subheader("Advantages")
        st.write("The Banker's algorithm stands out for its effective prevention of deadlocks, a critical advantage in ensuring system stability. By carefully allocating resources, it maintains a safe state, mitigating the risk of deadlock occurrence. This approach not only safeguards against system halts but also maximizes resource utilization. The algorithm allows resources to be allocated to processes whenever it is safe to do so, optimizing efficiency without compromising system integrity. Furthermore, the Banker's algorithm demonstrates flexibility in handling diverse resource requests from different processes. Its adaptable nature enables it to manage varying resource demands, enhancing its applicability across a range of scenarios. Overall, the Banker's algorithm offers a robust solution for deadlock prevention while promoting efficient resource utilization and accommodating the dynamic resource needs of processes.")
        st.subheader("DisAdvantages")
        st.write("While the Banker's algorithm effectively prevents deadlocks, it comes with several drawbacks. One notable issue is the potential for inefficient resource utilization, as the algorithm may hold resources even when they're not immediately needed by processes, leading to wastage. Additionally, the requirement for advanced knowledge of each process's maximum resource requirements may not always be practical. Moreover, the safety checks conducted after each resource allocation or release can introduce performance overhead, particularly in systems with numerous processes and resources. Finally, implementing the Banker's algorithm can be complex, especially in dynamic systems with varying resource allocation patterns. These limitations underscore the need for careful consideration of trade-offs when employing the Banker's algorithm in resource management.")
        
        #st.title("Banker's Algorithm Simulation")

        num_processes = st.number_input("Enter number of processes:", min_value=1, value=2)
        num_resources = st.number_input("Enter number of resources:", min_value=1, value=1)

        processes = [f"P{i}" for i in range(num_processes)]

        st.write("Enter available instances of resources:")
        avail = np.array([st.number_input(f"Available Resource {i + 1}:", min_value=0, step=1) for i in range(num_resources)])

        st.write("Enter maximum resource allocation for each process:")
        max_need = np.zeros((num_processes, num_resources), dtype=int)
        for i, process in enumerate(processes):
            st.write(f"For {process}:")
            for j in range(num_resources):
                max_need[i][j] = st.number_input(f"Maximum Resource {j + 1} for {process}:", min_value=0, step=1)

        st.write("Enter resource instance allocation for each process:")
        alloc = np.zeros((num_processes, num_resources), dtype=int)
        for i, process in enumerate(processes):
            st.write(f"For {process}:")
            for j in range(num_resources):
                alloc[i][j] = st.number_input(f"Allocated Resource {j + 1} for {process}:", min_value=0, step=1)

        if st.button("Simulate"):
            #choice = st.radio("Choose an action:", ("Check SAFE-state", "Request Resources"))
            #if choice == "Check SAFE-state":
            banker_algorithm(processes, avail, max_need, alloc)
            #elif choice == "Request Resources":
            #process_index = st.selectbox("Select a process:", processes)
                #resource_request = np.array([st.number_input(f"Request Resource {i + 1}:", min_value=0, step=1) for i in range(num_resources)])
                #if all(resource_request <= max_need[processes.index(process_index)] - alloc[processes.index(process_index)]):
                   # alloc[processes.index(process_index)] += resource_request
                    #avail -= resource_request
                    #st.write("Resource grant successfully.")
                   # banker_algorithm(processes, avail, max_need, alloc)
                #else:
                   # st.write("Resource request exceeds maximum available resources. Request denied.") 

  if 'Most Recently Used [MRU] Page Repelacment' in algo:
        st.subheader("MRU Page Replacement Algorithm")
        st.write("It is a page replacement policy used in operating systems to manage memory. In MRU, the page that has been accessed most recently is chosen for replacement when a new page needs to be brought into memory. This algorithm operates on the principle that pages that have been recently accessed are more likely to be accessed again in the near future.")
        st.subheader("Advantages")
        st.write("The MRU (Most Recently Used) algorithm prioritizes pages based on their recent access, making it advantageous for systems with frequent access patterns. This emphasis on temporal locality optimizes retrieval times by favoring recently used data. Additionally, its simplicity facilitates easy implementation compared to more complex page replacement policies, making it appealing for straightforward memory management. Overall, MRU offers a pragmatic solution for optimizing access performance, particularly in environments with predictable temporal access patterns.")
        st.subheader("DisAdvantages")
        st.write("While MRU (Most Recently Used) prioritizes recent access, it overlooks future access patterns, potentially leading to poor performance when patterns change rapidly. Moreover, its implementation demands extra memory and processing resources, especially in systems with numerous pages. Additionally, MRU's reliance on recent access history can contribute to thrashing when the working set exceeds memory capacity. These limitations underscore the need for alternative strategies that balance recent and future access considerations while minimizing overhead.")
        memory_size = st.number_input(label="Memory Size", min_value=1, max_value=10, value=3, step=1)
        page_references_input = st.text_input("Enter Page References (separated by commas)")
        page_references = [page.strip() for page in page_references_input.split(",")]  # Strip whitespace
        run_simulation = st.button("Run MRU Simulation")

        if run_simulation:
            try:
                page_faults, steps = mru_algorithm(page_references, memory_size)
                st.write(f"Number of Page Faults using MRU: {page_faults}")
                visualize_mru(steps)
            except ValueError:
                st.error("Invalid input. Please enter integers separated by commas.") 
                
  if 'Round Robin' in algo:
    st.subheader("Round Robin")
    st.write("RR is a CPU scheduling algorithm where each process is assigned a fixed time slot in a cyclic way, where each process gets equal share in time processing. RR is a hybrid model which is clock-driven. In this, CPU is shifted to the next process after fixed interval time, which is called time quantum/time slice. Time slice should be minimum, which is assigned for a specific task that needs to be processed. However, it may differ OS to OS.")
    st.subheader("Advantages")
    st.write("A round-robin scheduler generally employs time-sharing, giving each job a time slot or quantum. Each process get a chance to reschedule after a particular quantum time in this scheduling. All the jobs get a fair allocation of CPU.It deals with all process without any priority.This scheduling method does not depend upon burst time. That’s why it is easily implementable on the system.It gives the best performance in terms of average response time.")
    st.subheader("DisAdvantages")
    st.write("Gantt chart seems to come too big (if quantum time is less for scheduling.For Example:1 ms for big scheduling.)There is low throughput and context switches.Lower time quantum results in higher the context switching overhead in the system.Finding a correct time quantum is a quite difficult task in this system.If slicing time of OS is low, the processor output will be reduced.")
    st.title("Round Robin Scheduling Simulator")

    col1, col2, col3 = st.columns([1, 1, 2])  # Adjust column widths as needed

    n = col1.number_input("Enter the no. of processes:", min_value=1, step=1, value=1)
    quant = col2.number_input("Enter the quantum:", min_value=1, step=1, value=1)

    col3.write("Enter process details:")
    p = [Process() for _ in range(n)]
    for i in range(n):
        p[i].pos = col3.number_input(f"Process {i+1} number:", min_value=1, step=1, value=1)
        p[i].AT = col3.number_input(f"Arrival time of Process {i+1}:", min_value=0, step=1, value=0)
        p[i].BT = col3.number_input(f"Burst time of Process {i+1}:", min_value=1, step=1, value=1)
  # code will execute when the below button is prsses to avoid to much processing as streamlit reloads whenever any input value changes
  

    submit=st.button(label='Execute')
    if submit:
      # for algorithm in algo:
        st.title(algo)
        c = n
        time = 0
        index = -1
        s = [[-1] * 20 for _ in range(n)]
        b = [p[i].BT for i in range(n)]
        a = [p[i].AT for i in range(n)]
        tot_wt = 0
        tot_tat = 0

        gantt_data = []  # Data for Gantt chart

        while c != 0:
            mini = float("inf")
            flag = False

            for i in range(n):
                p[i].ST.append(time)
                pTime = time + 0.1
                if a[i] <= pTime and b[i] > 0:
                    if mini > a[i]:
                        index = i
                        mini = a[i]
                        flag = True

            if not flag:
                time += 1
                continue

            j = 0
            while s[index][j] != -1:
                j += 1

            if s[index][j] == -1:
                s[index][j] = time
                p[index].ST[j] = time

            if b[index] <= quant:
                time += b[index]
                b[index] = 0
            else:
                time += quant
                b[index] -= quant

            if b[index] > 0:
                a[index] = time + 0.1

            if b[index] == 0:
                c -= 1
                p[index].FT = time
                p[index].WT = p[index].FT - p[index].AT - p[index].BT
                tot_wt += p[index].WT
                p[index].TAT = p[index].BT + p[index].WT
                tot_tat += p[index].TAT

            # Append data for Gantt chart
            gantt_data.append({
                'Process': f'P{p[index].pos}',
                'Start': p[index].ST[j],
                'End': time
            })

        st.header("Gantt Chart")
        plot_gantt_chart(gantt_data)

        # Display results and averages
        st.header("Results")
        df_results = pd.DataFrame(columns=["Process Number", "Arrival Time", "Burst Time","Completion Time", "Turnaround Time", "Waiting Time"])
        for i in range(n):
            df_results.loc[i] = [i+1, p[i].AT, p[i].BT,p[i].TAT+p[i].AT,p[i].TAT,p[i].WT]
        st.table(df_results)

        avg_wt = tot_wt / n
        avg_tat = tot_tat / n

        st.write(f"The average wait time is : {avg_wt}")
        st.write(f"The average TurnAround time is : {avg_tat}")
        
  if 'Scan-Disk Scheduling' in algo:
        st.subheader("SCAN Disk Algorithm")
        st.write("The SCAN disk scheduling algorithm, also known as the Elevator algorithm is used to determine the order in which to service disk I/O requests. It works by servicing requests in a particular direction (e.g., from one end of the disk to the other) and then reversing direction when reaching the end until all requests have been serviced. The head of the disk moves in one direction servicing requests until it reaches the end, then it changes direction and continues servicing requests in the opposite direction.")
        st.subheader("Advantages")
        st.write("The SCAN disk scheduling algorithm offers several advantages. Firstly, it ensures fairness by servicing requests in the order they were received, without showing favoritism towards any specific region of the disk. Secondly, its unidirectional servicing approach typically leads to reduced average seek time compared to algorithms like FCFS. Finally, SCAN prevents starvation by consistently changing direction when reaching the disk's end, ensuring requests from all areas of the disk receive attention, thus averting potential prioritization issues based on track location.")
        st.subheader("DisAdvantages")
        st.write("The SCAN disk scheduling algorithm, while advantageous in some respects, also presents drawbacks. Firstly, there's a potential for increased latency, particularly for requests situated far from the current disk head position, especially if there are extended gaps between requests in certain areas of the disk. Secondly, its behavior can be unpredictable, especially when the direction of the disk head changes frequently, leading to varied response times for different requests. Lastly, SCAN may not be optimal for certain workloads, particularly those with heavy I/O activity concentrated in specific disk regions, where other scheduling algorithms might prove more efficient.")
  
        #st.title("SCAN Disk Scheduling Algorithm")
        st.write("\n")

        max_track_number = st.number_input("Enter the maximum track number:", min_value=1, step=1)
        num_requests = st.number_input("Enter the number of track requests:", min_value=1, step=1)
        track_range = range(max_track_number + 1)
    
        track_requests = []
        for i in range(num_requests):
            track_input = st.number_input(f"Enter Track {i + 1}:", min_value=0, max_value=max_track_number, step=1)
            if track_input in track_range:
                track_requests.append(track_input)
            else:
                st.error(f"Track number must be within the range of 0 to {max_track_number}.")

        head_position = st.number_input("Enter the current head position:", min_value=0, max_value=max_track_number, step=1)
        direction = st.selectbox("Select the direction to move:", ('Right', 'Left'))

        if st.button("Run SCAN Disk Algorithm"):
            traversed_tracks = []
            total_tracks_moved = 0

            while len(track_requests) > 0:
                traversed, moved = scan_disk(track_requests, head_position, direction, max_track_number)
                traversed_tracks += traversed
                total_tracks_moved += moved
                head_position = traversed[-1]
                track_requests = [track for track in track_requests if track not in traversed]

            st.write(f"Traversed tracks: {traversed_tracks}")
            st.write(f"Total tracks moved: {total_tracks_moved}")

            # Plotting the head traversal graph
            # Provided (x, y) values
            print("\n")
            print("\n")
            values = traversed_tracks

            # Sort values based on x-coordinate

            #sorted_values = sorted(values)

            # Extract x and y values
            x_values =traversed_tracks
            y_values =[]
            for i in range(1,len(traversed_tracks)+1):
                y_values.append(i)
        

            # Create a Streamlit app
            #st.title('Scatter Plot')

            # Display the provided values
            #st.write('Provided (x, y) values:')
            #st.write(values)

            # Create scatter plot
            st.write('SCAN Disk Algorithm Graph:')

            fig, ax = plt.subplots()
            ax.scatter(x_values, y_values, color='blue')
            ax.plot(x_values, y_values, linestyle='-', color='gray', alpha=1,)  # Add a line connecting the points
            plt.xticks(values,values)
            plt.figure(facecolor='#0e1117')
            ax.set_facecolor("#0e1117")
            

            ax.set_xlabel('Traversed Tracks')
            ax.set_ylabel('No. of Tracks')
            st.pyplot(fig)

  

if __name__=='__main__':
  app_layout()