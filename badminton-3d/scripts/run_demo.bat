@echo off
REM Chay toan bo video trong Input\ -> Output\ . Chay tu thu muc goc du an:  scripts\run_demo.bat
python -m badminton3d run || goto :err
echo Xong. Mo Output\_tong_hop\index.html (tong hop) hoac Output\^<video^>\report.html (tung video).
exit /b 0
:err
echo Co loi, xem thong bao phia tren.
exit /b 1
