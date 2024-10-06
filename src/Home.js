import React from "react";
import { FaHome, FaChartBar, FaMap, FaQuestionCircle, FaCog, FaChevronRight } from "react-icons/fa";
import { useNavigate } from "react-router-dom";

function Home() {
    const navigate = useNavigate();

    const handleClickHome = () => {
      navigate('/'); // Redireciona para /home
    };
  
    const handleClickDash = () => {
      navigate('/dashboard'); // Redireciona para /dashboard
    };
  
    const handleClickHealth = () => {
      navigate('/health'); // Redireciona para /health
    };

    const handleClickFaq = () => {
        navigate('/faq'); // Redireciona para /faq
    };
  
    return (
        <div className="h-screen flex" style={{ backgroundColor: '#171717E5', fontWeight: 500 }}>
          {/* Sidebar */}
          <aside className="min-w-80 h-screen flex flex-col" style={{ backgroundColor: '#232323', color: '#828282', paddingLeft: '1rem' }}>
            {/* Logo */}
            <div className="flex justify-center -mb-4 ">
              <img src="/images/logo.png" alt="Logo" className="h-40 w-auto" />
            </div>

            {/* Sidebar Items */}
            <ul className="space-y-8 ml-8">
              <li className="flex items-center justify-between w-full cursor-pointer" onClick={handleClickHome}>
                <div className="flex items-center space-x-4">
                  <FaHome className="text-green-500" size={24} />
                  <span style={{ color: '#1C6E14', fontSize: '18px' }}>Home</span>
                </div>
                <FaChevronRight className="text-gray-400" size={16} style={{ marginLeft: 'auto', marginRight: '25px' }} />
              </li>
              <li className="flex items-center justify-between w-full cursor-pointer" onClick={handleClickDash}>
                <div className="flex items-center space-x-4">
                  <FaChartBar className="text-gray-400" size={24} />
                  <span style={{ color: '#828282', fontSize: '18px' }}>Dashboard</span>
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

          {/* Main content with full-screen image and content */}
          <main className="relative w-full h-screen">
            {/* Background image */}
            <img 
              src="/images/background.png" // Caminho correto da imagem
              alt="Background"
              className="w-full h-full object-cover"
            />

            {/* Content with logo, text, and button */}
            <div className="absolute inset-0 flex flex-col items-center justify-center" style={{ backgroundColor: 'rgba(123, 206, 17, 0.15)' }}>
                {/* Logo */}
                <img src="/images/logo-white.png" alt="Logo" className="max-w-xl -mb-20" />
                
                {/* Text with black transparent background */}
                <div className="bg-black bg-opacity-70 text-white px-8 py-4 max-w-2xl rounded-lg" style={{fontFamily: 'Inter', fontWeight: 400}}>
                    <p>
                        Traditional agriculture faces challenges such as pests, climate variations, and high monitoring costs. 
                        To address these issues, FieldSentinel combines artificial intelligence with NASA hyperspectral data and 
                        vegetation indices, providing accurate insights into crop health.
                    </p>
                    <p className="mt-4">
                        Through a user-friendly and intuitive interface, even farmers without data expertise can determine the 
                        exact needs for water, pesticides, and fertilizers for each area of their farm. This information, integrated 
                        with automated drones/UAVs, enables the detection and treatment of problems before they escalate, ensuring 
                        bountiful and healthy harvests.
                    </p>
                </div>

                {/* Button */}
                <button 
                  className="mt-6 bg-green-600 text-white py-3 px-8 rounded transition-colors w-96 h-20"
                  style={{ backgroundColor: 'rgba(123, 206, 17, 0.6)', fontSize: '18px', fontWeight: '500', borderRadius: '4px', fontFamily: 'inter' }}
                  onClick={() => navigate('/learn-more')} // Define the desired action for the button
                >
                  Learn More
                </button>
            </div>
          </main>
        </div>
    );
}

export default Home;
