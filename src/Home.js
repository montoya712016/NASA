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

          {/* Main content with full-screen image and login form */}
          <main className="relative w-full h-screen">
            {/* Background image */}
            <img 
              src="/images/background.png" // Caminho correto da imagem
              alt="Background"
              className="w-full h-full object-cover"
            />

            {/* Login Form */}
            <div className="absolute inset-0 flex items-center justify-center" style={{backgroundColor: 'rgba(123, 206, 17, 0.15)'}}>

                <img src="/images/logo-white.png" alt="Logo" className="max-w-xl" />
            
              </div>
          </main>
        </div>
    );
}

export default Home;
