namespace CustomThreadPool
{
    using System;
    using System.Collections.Concurrent;
    using System.Collections.Generic;
    using System.Threading;

    /// <summary>
    /// A simple thread pool implementation.
    /// </summary>
    public class ThreadPool
    {
        private readonly List<Thread> _workers;
        private readonly BlockingCollection<Action> _taskQueue;
        private bool _isStopped = false;

        /// <summary>
        /// Initializes a new instance of the <see cref="ThreadPool"/> class with a specified number of threads.
        /// </summary>
        /// <param name="numThreads">The number of threads in the pool.</param>
        public ThreadPool(int numThreads)
        {
            _taskQueue = new BlockingCollection<Action>();
            _workers = new List<Thread>(numThreads);

            for (int i = 0; i < numThreads; i++)
            {
                var worker = new Thread(Work) { IsBackground = true };
                worker.Start();
                _workers.Add(worker);
            }
        }

        /// <summary>
        /// Enqueues a new task to the thread pool.
        /// </summary>
        /// <param name="task">The task to be executed by the thread pool.</param>
        public void EnqueueTask(Action task)
        {
            if (!_isStopped)
            {
                _taskQueue.Add(task);
            }
        }

        /// <summary>
        /// Executes tasks from the queue in a loop until the pool is stopped.
        /// </summary>
        private void Work()
        {
            foreach (var task in _taskQueue.GetConsumingEnumerable())
            {
                if (_isStopped)
                {
                    return;
                }

                task();
            }
        }

        /// <summary>
        /// Shuts down the thread pool and waits for all threads to complete their current tasks.
        /// </summary>
        public void Shutdown()
        {
            _taskQueue.CompleteAdding();
            
            foreach (var worker in _workers)
            {
                worker.Join();
            }
            
            _isStopped = true;
        }
    }
}
