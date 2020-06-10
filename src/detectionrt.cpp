int detectNet::Detect( float* rgba, uint32_t width, uint32_t height, Detection* detections, uint32_t overlay )
{
	if( !rgba || width == 0 || height == 0 || !detections )
	{		
		return -1;
	}	

	if( IsModelType(MODEL_UFF) )
	{
		if( CUDA_FAILED(cudaPreImageNetNormBGR((float4*)rgba, width, height, mInputCUDA, mWidth, mHeight,
										  make_float2(-1.0f, 1.0f), GetStream())) )
		{			
			return -1;
		}
	}
	else if( IsModelType(MODEL_ONNX) )
	{		
		if( CUDA_FAILED(cudaPreImageNetNormMeanRGB((float4*)rgba, width, height, mInputCUDA, mWidth, mHeight, 
										   make_float2(0.0f, 1.0f), 
										   make_float3(0.485f, 0.456f, 0.406f),
										   make_float3(0.229f, 0.224f, 0.225f), 
										   GetStream())) )
		{			
			return false;
		}
	}
	else
	{
		if( mMeanPixel != 0.0f )
		{
			if( CUDA_FAILED(cudaPreImageNetMeanBGR((float4*)rgba, width, height, mInputCUDA, mWidth, mHeight,
										  make_float3(mMeanPixel, mMeanPixel, mMeanPixel), GetStream())) )
			{				
				return -1;
			}
		}
		else
		{
			if( CUDA_FAILED(cudaPreImageNetBGR((float4*)rgba, width, height, mInputCUDA, mWidth, mHeight, GetStream())) )
			{				
				return -1;
			}
		}
	}
	
	// process with TensorRT
	void* inferenceBuffers[] = { mInputCUDA, mOutputs[0].CUDA, mOutputs[1].CUDA };
	
	if( !mContext->execute(1, inferenceBuffers) )
	{		
		return -1;
	}

	// post-processing / clustering
	int numDetections = 0;

	if( IsModelType(MODEL_UFF) )
	{		
		const int rawDetections = *(int*)mOutputs[OUTPUT_NUM].CPU;
		const int rawParameters = DIMS_W(mOutputs[OUTPUT_UFF].dims);

		// filter the raw detections by thresholding the confidence
		for( int n=0; n < rawDetections; n++ )
		{
			float* object_data = mOutputs[OUTPUT_UFF].CPU + n * rawParameters;

			if( object_data[2] < mCoverageThreshold )
				continue;

			detections[numDetections].Instance   = numDetections; 
			detections[numDetections].ClassID    = (uint32_t)object_data[1];
			detections[numDetections].Confidence = object_data[2];
			detections[numDetections].Left       = object_data[3] * width;
			detections[numDetections].Top        = object_data[4] * height;
			detections[numDetections].Right      = object_data[5] * width;
			detections[numDetections].Bottom	  = object_data[6] * height;

			if( detections[numDetections].ClassID >= mNumClasses )
			{				
				detections[numDetections].ClassID = 0;
			}

			if( strcmp(GetClassDesc(detections[numDetections].ClassID), "void") == 0 )
				continue;

			numDetections += clusterDetections(detections, numDetections);
		}

		// sort the detections by confidence value
		sortDetections(detections, numDetections);
	}
	else if( IsModelType(MODEL_ONNX) )
	{
		float* coord = mOutputs[0].CPU;

		coord[0] = ((coord[0] + 1.0f) * 0.5f) * float(width);
		coord[1] = ((coord[1] + 1.0f) * 0.5f) * float(height);
		coord[2] = ((coord[2] + 1.0f) * 0.5f) * float(width);
		coord[3] = ((coord[3] + 1.0f) * 0.5f) * float(height);		

		detections[numDetections].Instance   = numDetections;
		detections[numDetections].ClassID    = 0;
		detections[numDetections].Confidence = 1;
		detections[numDetections].Left       = coord[0];
		detections[numDetections].Top        = coord[1];
		detections[numDetections].Right      = coord[2];
		detections[numDetections].Bottom	  = coord[3];	
	
		numDetections++;
	}
	else
	{
		// cluster detections
		numDetections = clusterDetections(detections, width, height);
	}
	
	// render the overlay
	if( overlay != 0 && numDetections > 0 )
	{
		if( !Overlay(rgba, rgba, width, height, detections, numDetections, overlay) )
			printf("detectNet::Detect() -- failed to render overlay\n");
	}
	
	// wait for GPU to complete work			
	CUDA(cudaDeviceSynchronize());

	// return the number of detections
	return numDetections;
}