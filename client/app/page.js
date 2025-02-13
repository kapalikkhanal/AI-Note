// app/page.js
'use client'
import { useState } from 'react'
import { PlusCircle } from 'lucide-react'

export default function Home() {
  const [tasks, setTasks] = useState([])
  const [newTask, setNewTask] = useState('')
  const [sortBy, setSortBy] = useState('priority') // 'priority' or 'date'

  const priorityColors = {
    High: 'bg-red-100 border-red-200',
    Medium: 'bg-yellow-100 border-yellow-200',
    Low: 'bg-green-100 border-green-200'
  }

  const priorityOrder = {
    High: 1,
    Medium: 2,
    Low: 3
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!newTask.trim()) return

    try {
      const response = await fetch('http://localhost:5000/api/predict', {
        method: 'POST',
        headers: {
          'Authorization': 'Bearer 1234',
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ task: newTask })
      })

      const data = await response.json()
      if (data.status === 'success') {
        setTasks([...tasks, {
          ...data,
          id: Date.now() // Add a unique ID for each task
        }])
        setNewTask('')
      }
    } catch (error) {
      console.error('Error adding task:', error)
    }
  }

  const getSortedTasks = () => {
    return [...tasks].sort((a, b) => {
      if (sortBy === 'priority') {
        return priorityOrder[a.priority] - priorityOrder[b.priority]
      } else {
        return new Date(b.timestamp) - new Date(a.timestamp)
      }
    })
  }

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-4xl mx-auto px-4">
        <h1 className="text-3xl font-bold text-gray-900 mb-8">Task Manager</h1>
        
        {/* Add Task Form */}
        <form onSubmit={handleSubmit} className="mb-8">
          <div className="flex gap-4">
            <input
              type="text"
              value={newTask}
              onChange={(e) => setNewTask(e.target.value)}
              placeholder="Enter new task..."
              className="flex-1 text-black rounded-lg border border-gray-300 px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <button
              type="submit"
              className="flex items-center gap-2 bg-blue-500 text-white px-6 py-2 rounded-lg hover:bg-blue-600 transition-colors"
            >
              <PlusCircle className="w-5 h-5" />
              Add Task
            </button>
          </div>
        </form>

        {/* Sort Controls */}
        <div className="mb-6">
          <label className="text-sm font-medium text-gray-900">Sort by: </label>
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value)}
            className="ml-2 rounded-md border border-gray-300 px-3 py-1"
          >
            <option className='text-black' value="priority">Priority</option>
            <option className='text-black' value="date">Date</option>
          </select>
        </div>

        {/* Task List */}
        <div className="space-y-4">
          {getSortedTasks().map((task) => (
            <div
              key={task.id}
              className={`rounded-lg border p-4 ${priorityColors[task.priority]} transition-all`}
            >
              <div className="flex justify-between items-start">
                <div>
                  <h3 className="text-lg font-medium text-gray-900">{task.task}</h3>
                  <p className="text-sm text-gray-600">
                    Priority: {task.priority} | Confidence: {task.confidence}
                  </p>
                  <p className="text-sm text-gray-500">
                    {new Date(task.timestamp).toLocaleString()}
                  </p>
                </div>
              </div>
            </div>
          ))}
          
          {tasks.length === 0 && (
            <p className="text-center text-gray-500 py-8">
              No tasks yet. Add your first task above!
            </p>
          )}
        </div>
      </div>
    </div>
  )
}