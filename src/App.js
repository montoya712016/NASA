import React from "react";
import { FaHome, FaChartBar, FaMap, FaQuestionCircle, FaCog, FaChevronRight } from "react-icons/fa";
import { useNavigate } from 'react-router-dom';


function App() {
  const navigate = useNavigate();

  const handleClickHome = () => {
    navigate('/'); // Redireciona para /home
  };

  const handleClickDash = () => {
    navigate('/dashboard'); // Redireciona para /home
  };

  const handleClickHealth = () => {
    navigate('/health'); // Redireciona para /home
  };

  const handleClickFaq = () => {
    navigate('/faq'); // Redireciona para /home
  };


  return (
    <div className="min-h-screen flex" style={{ backgroundColor: '#171717E5', fontWeight: 500 }}>
      {/* Sidebar */}
      <aside className="w-80 flex flex-col" style={{ backgroundColor: '#232323', color: '#828282', paddingLeft: '1rem' }}>
        {/* Logo */}
        <div className="flex justify-center -mb-4 ">
          <img src="/images/logo.png" alt="Logo" className="h-40 w-auto" />
        </div>

        {/* Sidebar Items */}
        <ul className="space-y-8 ml-8">
          <li className="flex items-center justify-between w-full cursor-pointer" onClick={handleClickHome}>
            <div className="flex items-center space-x-4">
              <FaHome className="text-gray-400" size={24} />
              <span style={{ color: '#828282', fontSize: '18px' }}>Home</span>
            </div>
            <FaChevronRight className="text-gray-400" size={16} style={{ marginLeft: 'auto', marginRight: '25px' }} />
          </li>
          <li className="flex items-center justify-between w-full cursor-pointer" onClick={handleClickDash}>
            <div className="flex items-center space-x-4">
              <FaChartBar className="text-green-500" size={24} />
              <span style={{ color: '#1C6E14', fontSize: '18px' }}>Dashboard</span>
            </div>
            <FaChevronRight className="text-gray-400" size={16} style={{ marginLeft: 'auto', marginRight: '25px' }} />
          </li>
          <li className="flex items-center justify-between w-full cursor-pointer" onClick={handleClickHealth}>
            <div className="flex items-center space-x-4">
              <FaMap className="text-gray-400" size={24} />
              <span style={{ color: '#828282', fontSize: '18px' }}>Health Map</span>
            </div>
            <FaChevronRight className="text-gray-400" size={16} style={{ marginLeft: 'auto', marginRight: '25px' }} />
          </li>
          <li className="flex items-center justify-between w-full cursor-pointer" onClick={handleClickFaq}>
            <div className="flex items-center space-x-4">
              <FaQuestionCircle className="text-gray-400" size={24} />
              <span style={{ color: '#828282', fontSize: '18px' }}>Help</span>
            </div>
            <FaChevronRight className="text-gray-400" size={16} style={{ marginLeft: 'auto', marginRight: '25px' }} />
          </li>
        </ul>
      </aside>

      {/* Main Content */}
      <main className="flex-1 p-6">
        {/* Dashboard Title */}
        <h1 className="text-4xl mb-6" style={{ fontFamily: 'Poppins', fontWeight: 'bold', color: '#F2F2F2' }}>Dashboard</h1>

        <div className="grid grid-cols-5 gap-6">
          {/* Left Column: Region Map and Culture Health */}
          <div className="col-span-3">
            {/* Full Region Map Panel */}
            <div className="bg-[#232323] shadow-lg p-6 text-green-400 mb-6 rounded-lg">
              <div className="flex items-start">
                <h2 className="text-3xl " style={{ color: '#7BCE11', fontFamily: 'Inter', fontWeight: '400' }}>Region Map</h2>
                <p className="text-lg  ml-36" style={{ color: '#F2F2F2', marginTop: '8px', fontFamily: 'Inter', fontWeight: '400' }}>Piracicaba, São Paulo</p>
              </div>
              <div className="flex mt-4">
                <img src="/images/piracicaba.png" alt="Map" className="w-2/5 rounded-md" />
                <div className="ml-6 text-gray-100 flex flex-col justify-center space-y-2">
                  <div className="grid grid-cols-2 gap-x-4 gap-y-1">
                    <p>Current temperature:</p>
                    <p style={{ color: '#1C6E14', textAlign: 'right' }}>22ºC</p>
                    <p>Precipitation:</p>
                    <p style={{ color: '#1C6E14', textAlign: 'right' }}>69%</p>
                    <p>Wind:</p>
                    <p style={{ color: '#1C6E14', textAlign: 'right' }}>11 km/h</p>
                    <p>Humidity:</p>
                    <p style={{ color: '#1C6E14', textAlign: 'right' }}>95% UR</p>
                  </div>
                </div>
              </div>
            </div>

            {/* First Culture Health */}
            <div className="bg-[#232323] shadow-lg rounded-lg p-4">
              <h2 className="text-3xl font-bold" style={{ color: '#7BCE11', fontFamily: 'Inter', fontWeight: '400' }}>Culture Health</h2>
              <div className="flex justify-center">
                <img src="/images/chart_health.png" alt="Culture Health Chart" className="rounded-md w-4/5" />
              </div>
              <p className="text-center text-gray-400 mt-2">Latest vegetation stress analysis</p>
            </div>
          </div>

          {/* Right Column: Weather Data */}
          <div className="bg-[#232323] shadow-lg rounded-lg p-6 col-span-2 flex flex-col justify-between">
            <h2 className="text-3xl font-bold mb-6" style={{ color: '#7BCE11', fontFamily: 'Inter', fontWeight: '400' }}>Weather Data</h2>
            <div className="flex items-center mb-4">
              <div className="text-gray-100 mr-4">
                <p className="font-bold">PERIOD</p>
                <p>01/09/2024 - 30/09/2024</p>
                <p className="font-bold mt-2">PRECIPITATION</p>
                <p>Rain: <span className="text-blue-400 font-semibold">25 mm</span></p>
              </div>
              <img src="/images/weather1.png" alt="Weather Data 1" className="w-3/5 rounded-md" />
            </div>
            <div className="flex items-center mb-4">
              <div className="text-gray-100 mr-4">
                <p className="font-bold">PERIOD</p>
                <p>01/09/2024 - 30/09/2024</p>
                <p className="font-bold mt-2">TEMPERATURE</p>
                <p>Average: <span className="text-red-400 font-semibold">18 to 26 ºC</span></p>
              </div>
              <img src="/images/weather2.png" alt="Weather Data 2" className="w-3/5 rounded-md" />
              
            </div>
            <p className="text-center text-gray-400">Tap on the images to view more information.</p>
            
            <div className="flex justify-center mt-4">
              <button
                className="bg-green-600 text-white text-lg px-8 py-4 rounded-lg shadow-lg"
                style={{ border: '2px solid #000', boxShadow: '0px 4px 8px rgba(0, 0, 0, 0.3)' }}
              >
                Weather conditions are favorable!
              </button>
            </div>
          </div>
        </div>

        {/* Final Row: Culture Health and Collected Images */}
        <div className="grid grid-cols-5 gap-6 mt-6">
          {/* Second Culture Health */}
          <div className="bg-[#232323] shadow-lg rounded-lg p-4 col-span-3">
            <h2 className="text-3xl font-bold" style={{ color: '#7BCE11', fontFamily: 'Inter', fontWeight: '400' }}>Culture Health</h2>
            <div className="flex justify-center">
              <img src="/images/second_chart.png" alt="Second Culture Health Chart" className="rounded-md w-3/5" />
            </div>
            <p className="text-center text-gray-400 mt-2">Crop health analysis over months</p>
          </div>

          {/* Collected Images */}
          <div className="bg-[#232323] shadow-lg rounded-lg p-4 col-span-2">
            <h2 className="text-3xl font-bold" style={{ color: '#7BCE11', fontFamily: 'Inter', fontWeight: '400' }}>Collected Images</h2>
            <div className="flex justify-center">
              <img src="/images/images.png" alt="Collected Image" className="rounded-md mt-4 w-4/5" />
            </div>
            <p className="text-center text-gray-400 mt-2">Tap on the images to view more information.</p>
          </div>
        </div>
      </main>
    </div>
  );
}

export default App;
