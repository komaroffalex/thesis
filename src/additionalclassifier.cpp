bool isTrolley(cv::Mat& templateFrame, cv::Mat& compareFrame, const float ratio_thresh, bool isUseContourFilter, bool isUseLaplacian, int keypointsThreashold) {
    if (isUseLaplacian) {
        cv::Laplacian(templateFrame, templateFrame, CV_8U);
        cv::Laplacian(compareFrame, compareFrame, CV_8U);
    }

    if (isUseContourFilter) {
        cv::Mat cannyOutputTemplate;
        cv::Mat cannyOutputCompare;
        vector<vector<cv::Point> > contoursTemplate;
        vector<cv::Vec4i> hierarchyTemplate;
        vector<vector<cv::Point> > contoursCompare;
        vector<cv::Vec4i> hierarchyCompare;

        // Detect edges using canny
        Canny(templateFrame, cannyOutputTemplate, 100, 100 * 2, 3);
        Canny(compareFrame, cannyOutputCompare, 100, 100 * 2, 3);
        // Find contours
        findContours(cannyOutputTemplate, contoursTemplate, hierarchyTemplate, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
        findContours(cannyOutputCompare, contoursCompare, hierarchyCompare, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

        // Draw contours
        cv::Mat drawingTemplate = cv::Mat::zeros(cannyOutputTemplate.size(), CV_8UC1);

        for (int i = 0; i < contoursTemplate.size(); i++)
        {
            cv::Scalar color = cv::Scalar(255, 0, 255);
            drawContours(drawingTemplate, contoursTemplate, i, color, 1, 8, hierarchyTemplate, 0, cv::Point());
        }

        cv::Mat drawingCompare = cv::Mat::zeros(cannyOutputCompare.size(), CV_8UC1);
        for (int i = 0; i < contoursCompare.size(); i++)
        {
            cv::Scalar color = cv::Scalar(255, 0, 255);
            drawContours(drawingCompare, contoursCompare, i, color, 1, 8, hierarchyCompare, 0, cv::Point());
        }
        
        templateFrame = drawingTemplate;
        compareFrame = drawingCompare;
    }

    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    vector<cv::KeyPoint> regionsTemplate, regionsCompare;
    vector<cv::Rect> orb_bboxTemplate, orb_bboxCompare;
    cv::Mat descriptorsTemplate, descriptorsCompare;
    orb->detectAndCompute(templateFrame, cv::noArray(), regionsTemplate, descriptorsTemplate);
    orb->detectAndCompute(compareFrame, cv::noArray(), regionsCompare, descriptorsCompare);

    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_SL2);
    std::vector< std::vector<cv::DMatch> > knn_matches;
    std::vector< cv::DMatch> matches;
    matcher->knnMatch(descriptorsTemplate, descriptorsCompare, knn_matches, 2);
    
    std::vector<cv::DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++)
    {
        if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
        {
            good_matches.push_back(knn_matches[i][0]);
        }
    }
    //-- Draw matches
    cv::Mat img_matches;
    drawMatches(templateFrame, regionsTemplate, compareFrame, regionsCompare, good_matches, img_matches, cv::Scalar::all(-1),
        cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    //-- Show detected matches    
    imshow("Good Matches", img_matches);
    cv::waitKey();    
    return (good_matches.size() >= keypointsThreashold) ? true : false;
}