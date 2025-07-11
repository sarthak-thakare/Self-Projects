#include<bits/stdc++.h>
#include <fstream>
#include <sstream> 
#include <iomanip>
#include <ctime>
#include <csignal>

using namespace std;

#define GREEN "\033[32m"
#define RED "\033[31m"
#define YELLOW "\033[33m"
#define RESET "\033[0m"

struct Task{
    int Id;
    string title;
    bool completed;
    string due_date;   //YYYY-MM-DD
    int priority;   //1(highest) to 5(lowest)

    Task(string Title){
        this->Id=0;
        this->completed=false;
        this->title=Title;
    }
};

vector<Task>tasks;
int next_id=1;
string storage_file = "tasks.csv";


vector<string>split(string& line,char delimiter){
    vector<string>tokens;
    string token;
    istringstream ss(line);
    while(getline(ss,token,delimiter)){
        tokens.push_back(token);
    }
    return tokens;
}
string toLower(string s) {
    transform(s.begin(), s.end(), s.begin(), ::tolower);
    return s;
}

bool isValidDate(string date){
    if (date.length() != 10) return false;
    if (date[4] != '-' || date[7] != '-') return false;
    for (int i = 0; i < date.size(); ++i) {
        if (i == 4 || i == 7) continue;
        if (!isdigit(date[i])) return false;
    }
    int year = stoi(date.substr(0, 4));
    int month = stoi(date.substr(5, 2));
    int day = stoi(date.substr(8, 2));
    if (month < 1 || month > 12) return false;
    int max_day;
    switch (month) {
        case 1: case 3: case 5: case 7: case 8: case 10: case 12:
        max_day = 31; break;
        case 4: case 6: case 9: case 11:
        max_day = 30; break;
        case 2:
        if ((year % 4 == 0 && year % 100 != 0) || (year % 400 == 0))
        max_day = 29;
        else
        max_day = 28;
        break;
        default:
        return false;
    }
    if (day < 1 || day > max_day) return false;
    return true;
}

void SaveTasks(){
    // if (tasks.empty()) return;
    ofstream out(storage_file,ios::trunc);
    out<<"id,title,due_date,priority,completed\n";
    sort(tasks.begin(),tasks.end(),[](const Task &a, const Task &b){
        if(a.completed!=b.completed){
            return a.completed<b.completed;
        }
        if(a.due_date!=b.due_date){
            return a.due_date<b.due_date;
        }
        return a.priority<b.priority;
    });
    for(auto task:tasks){
        out<<task.Id<<","<<task.title<<","<<task.due_date<<","<<task.priority<<","<<task.completed<<"\n";
    }
    // cout<<GREEN<<"Tasks saved to "<<storage_file<<RESET<<endl;
    out.close();
}

void handleSigInt(int sig){
    SaveTasks();
    cout << "\n" << YELLOW << "Interrupt signal (" << sig << ") received. Saving tasks before exit..." << RESET<<flush;
    exit(sig);
}

//  Add new task to tasks vector
void handleAdd(string title){
    auto parts=split(title,' ');
    int n=parts.size();
    if(n<3){
        cout<<RED<<"Add command should have space separated due_date and priority at end\n"<<RESET;
        return;
    }
    string title_here="";
    for(int i=0;i<n-2;i++){
        title_here+=parts[i];
        if(i==n-3)break;
        title_here+=" ";
    }
    int prior=stoi(parts[n-1]);
    string date=parts[n-2];
    if(prior<1 || prior>5){
        cout<<RED<<"Priority should be between 1 to 5.\n"<<RESET;
        return;
    }
    if(!isValidDate(date)){
        cout<<RED<<"Due date should be YYYY-MM-DD.\n"<<RESET;
        return;
    }
    Task temp(title_here);
    temp.Id=next_id++;
    temp.priority=prior;
    temp.due_date=date;
    tasks.push_back(temp);
    cout<<GREEN<<"Task added."<<RESET<<endl;
    SaveTasks();
}

// Print Pending tasks
void handleList(){
    // sort(tasks.begin(),tasks.end(),[](const Task &a, const Task &b){
    // if(a.completed!=b.completed){
    //         return a.completed<b.completed;
    //     }
    //     if(a.due_date!=b.due_date){
    //         return a.due_date<b.due_date;
    //     }
    //     return a.priority<b.priority;
    // });
    // cout << left << setw(5) << "ID" << setw(30) << "Title" << setw(15) << "Due Date" << setw(10) << "Priority" << endl;
    
    cout << GREEN;
    cout << "+--------+------------------------------------------+------------+----------+\n";
    cout << "| ID     |                   Title                  | Due Date   | Priority |\n";
    cout << "+--------+------------------------------------------+------------+----------+\n";

    for (auto &task : tasks) {
        // string status;
        if (!task.completed) {
            cout << "| " << setw(6) <<left<< task.Id << " | "
                 << setw(40) << left << task.title << " | "
                 << setw(10) << task.due_date << " | "
                 << setw(8) << task.priority <<" | \n";
                }
        
    }

    cout << "+--------+------------------------------------------+------------+----------+\n"<<RESET;

}

// Mark a task completed
void markTaskCompleted(int id){
    for(auto& task:tasks){
        if(task.Id==id){
            task.completed=true;
            cout<<GREEN<<"Task "<<id<<" completed.\n"<<RESET;
            SaveTasks();
            return;
        }
    }
    cout<<RED<<"Task not found.\n"<<RESET;
}

// Show history of completed tasks
void showCompletedTasks(){
    cout << left << setw(5) << "ID" << setw(30) << "Title" << setw(15) << "Due Date" << setw(10) << "Priority" << endl;
    for(auto& task:tasks){
        if(task.completed==true){
            cout << left << setw(5) << task.Id << setw(30) << task.title << setw(15) << task.due_date << setw(10) << task.priority << endl;
        }
    }
    cout<<GREEN<<"These tasks are completed.\n"<<RESET;
}

//delete task from task vector
void deleteTask(int id){
    if(id==-1){
        cout<<"Are you sure you want to delete ALL tasks? (yes/no): ";
        string confirm;
        getline(cin, confirm);
        if (toLower(confirm) == "yes") {
            tasks.clear();
            cout<<GREEN<<"All tasks deleted.\n"<<RESET;
            SaveTasks();
        } else {
            cout<<"Deletion cancelled.\n";
        }
        return;
    }
    for(auto it=tasks.begin();it!=tasks.end();it++){
        if(it->Id==id){
            tasks.erase(it);
            cout<<GREEN<<"Task "<<id<<" deleted.\n"<<RESET;
            SaveTasks();
            return;
        }
    }
    cout<<RED<<"Task not found.\n"<<RESET;
}

// edit a task
void editTask(string s){
    auto parts=split(s,' ');
    int n=parts.size();
    if(n<3){
        cout<<RED<<"Enter edit <id> <field> <new_value>\n"<<RESET;
        return;
    }
    int id=stoi(parts[0]);
    string field=parts[1];
    string new_value="";
    for(int i=2;i<n;i++){
        new_value+=parts[i];
        if(i==n-1)break;
        new_value+=" ";
    }
    int new_priority = 0;
    if (field == "priority") {
        new_priority = stoi(new_value);
        if (new_priority < 1 || new_priority > 5) {
            cout<<RED<<"Priority must be between 1 and 5.\n"<<RESET;
            return;
        }
    }
    for(auto& task:tasks){
        if(task.Id==id){
            if (field == "title") {
                task.title = new_value;
            } else if (field == "due_date") {
                if (!isValidDate(new_value)) {
                    cout<<RED<<"Invalid date format. Use YYYY-MM-DD.\n"<<RESET;
                    return;
                }
                task.due_date = new_value;
                SaveTasks();
            } else if (field == "priority") {
                task.priority = new_priority;
                SaveTasks();
            } else {
                cout<<RED<<"Field must be one of: title, due_date, priority\n"<<RESET;
                return;
            }
            cout<<GREEN<<"Task "<<id<<" updated successfully.\n"<<RESET;
            return;
        }
    }
    cout<<RED<<"Task with ID "<<id<<" not found.\n"<<RESET;
}


void showHelp() {
    cout << GREEN << "Available Commands:\n" << RESET;
    cout << "  add <title> <due_date YYYY-MM-DD> <priority 1-5>\n";
    cout << "  list                          - List all pending tasks\n";
    cout << "  complete <id>                - Mark a task as completed\n";
    cout << "  delete <id> or delete -1     - Delete a task or all tasks\n";
    cout << "  search <keyword>             - Search for tasks by title\n";
    cout << "  edit <id> <field> <new_value> - Edit task field\n";
    cout << "  history                      - Show completed tasks\n";
    cout << "  reminder                     - Show Upcoming pending tasks for today and tomorrow\n";
    cout << "  help                         - Show this help message\n";
    cout << "  export                       - export to tasks.md\n";
    cout << "  exit                         - Save and exit\n";
}


string getTodayDate(){
    time_t t=time(nullptr);
    tm* now=localtime(&t);
    char buf[11];
    strftime(buf,sizeof(buf),"%Y-%m-%d", now);
    return string(buf);
}

string getTomorrowDate(){
    time_t t=time(nullptr);
    tm* now=localtime(&t);
    now->tm_mday+=1;
    mktime(now);
    char buf[11];
    strftime(buf,sizeof(buf),"%Y-%m-%d", now);
    return string(buf);
}

// show tasks whose due-date is today or tomorrow
void showReminders(){
    string today=getTodayDate();
    string tomorrow=getTomorrowDate();

    vector<Task>todaytasks;
    vector<Task>tomorrowtasks;
    bool found_today=false;
    bool found_tomorrow=false;

    for(auto &task : tasks){
        if(task.due_date==today){
            todaytasks.push_back(task);
            found_today=true;
        }
        if(task.due_date==tomorrow){
            tomorrowtasks.push_back(task);
            found_tomorrow=true;
        }  
    }
    if(found_today || found_tomorrow){
         cout << GREEN << "\n Upcoming Tasks\n" << RESET;
        if(found_today){
          cout<< RED <<" TODAY:\n"<<RESET;
          for(auto task:todaytasks){
            cout << " [" << task.Id << "] " << task.title << " Due: " << task.due_date
                 << "  Priority: " << task.priority << endl;
          }
        }
        if(found_tomorrow){
          cout<< RED <<" TOMORROW:\n"<<RESET;
          for(auto task:tomorrowtasks){
            cout << " [" << task.Id << "] " << task.title << "  Due: " << task.due_date
                 << "  Priority: " << task.priority << endl;
          }
        }
    }
    else{
         cout << YELLOW<<" No urgent tasks for today or tomorrow!\n"<<RESET;
    }

}

// Load current tasks from csv
void LoadTasks(){
    ifstream file(storage_file);
    string header;
    getline(file,header);
    string line;
    while(getline(file,line)){
        auto fields=split(line, ',');
        if (fields.size() < 5) continue;
        Task t("");
        t.Id=stoi(fields[0]);
        t.title=fields[1];
        t.due_date=fields[2];
        t.priority=stoi(fields[3]);
        t.completed=stoi(fields[4]);
        tasks.push_back(t);
        next_id=max(next_id,t.Id+1);
    }
}

// Save changes made in tasks vector to csv

void exportToMarkdown(){
    ofstream out("tasks.md");
    out <<"Task List\n\n";
    out << "+--------+------------------------------------------+------------+----------+------------+\n";
    out << "| ID     |                   Title                  | Due Date   | Priority | Status     |\n";
    out << "+--------+------------------------------------------+------------+----------+------------+\n";

    for (auto &task : tasks) {
        string status;
        if (!task.completed) {status="Pending";}
        else status="Completed";
            out << "| " << setw(6) <<left<< task.Id << " | "
                 << setw(40) << left << task.title << " | "
                 << setw(10) << task.due_date << " | "
                 << setw(8) << task.priority <<" | "
                 << setw(10) << status<<" |\n";
        
    }

    out << "+--------+------------------------------------------+------------+----------+------------+\n";
        out.close();
    cout << GREEN << " Exported tasks to tasks.md \n" << RESET;

}

// search task which has given keyword
void search(string keyword){
    bool found = false;
    for(auto task:tasks){
        if(toLower(task.title).find(toLower(keyword))!=string::npos){
            found = true;
            cout << "[" << task.Id << "]," << task.title <<","<<task.due_date<<","<<task.priority;
            if(!task.completed)cout<<",PENDING\n";
            else cout<<",COMPLETED\n";
        }
    }
    if (!found) cout<<RED<<"No task found.\n"<<RESET;
}

int main(){
    signal(SIGINT,handleSigInt);

    string command;
    cout<<GREEN<<"CLI Task Manager >"<<RESET;
    LoadTasks();
    while(getline(cin,command)){
        if(command.rfind("add ",0)==0){
            handleAdd(command.substr(4));
        }
        else if(command.rfind("complete ",0)==0){
            markTaskCompleted(stoi(command.substr(9)));
        }
        else if(command.rfind("delete ",0)==0){
            deleteTask(stoi(command.substr(7)));
        }
        else if(command.rfind("search ",0)==0){
            search(command.substr(7));
        }
        else if(command.rfind("edit ",0)==0){
            editTask(command.substr(5));
        }
        else if(command=="list"){
            handleList();
        }
        else if(command=="reminder"){
            showReminders();
        }
        else if(command=="history"){
            showCompletedTasks();
        }
        else if(command=="help"){
            showHelp();
        }
        else if(command=="export"){
            exportToMarkdown();
        }
        else if(command=="exit"){
            break;
        }
        else{
            cout<<RED<<"Unknown Command.\n"<<RESET;
        }
        cout<<GREEN<<"CLI Task Manager >"<<RESET;
    }
    SaveTasks();
    cout<<GREEN<<"Goodbye !\n"<<RESET;
}
