#ifndef QUEUE_HASH_GPU_CU
#define QUEUE_HASH_GPU_CU

///
///		Fixed Memory Rnaged Queue
///

//	mimic the template of CPU code
#define Type short

//
//	looped fixed-range queue.
//	_nFrontInx point to the head, and _nEndInx points to the first NULL element
//
struct range_queue_gpu
{

	__device__
	range_queue_gpu(Type *pT, unsigned count)
	{
		_data  = pT;
		_nFrontInx = 0;
		_nEndInx = 0;
		_count = count;
		_currCount = 0;
	}

	__device__
	void pop()
	{
		if(size() > 0)
		{
			_nFrontInx ++;
			_nFrontInx %= _count;
			_currCount --;
		}
	}

	__device__
	void push(Type &t)
	{
		if(size() < _count)
		{
			*(_data + _nEndInx) = t;
			_nEndInx += 1; _nEndInx %= _count;
			_currCount ++;
		}
	}

	__device__
	Type &front()
	{	return *(_data + _nFrontInx); }

	__device__
	unsigned size()
	{	
		return _currCount;
	}

private:

	Type *_data;
	unsigned _nFrontInx;
	unsigned _nEndInx;

	unsigned _currCount;
	unsigned _count;
};

//	the hash_map could be a directly indexed array

//

#endif